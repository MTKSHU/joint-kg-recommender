import logging
import gflags
import sys
import os
import json
from tqdm import tqdm
tqdm.monitor_iterval=0
import math
import multiprocessing
import time

import torch
import torch.nn as nn
from torch.autograd import Variable as V

from jTransUP.models.base import get_flags, flag_defaults, load_data
from jTransUP.models.base import init_model
from jTransUP.utils.trainer import ModelTrainer
from jTransUP.utils.misc import Accumulator, getPerformance, MyTopNRecEvalProcess
from jTransUP.utils.misc import to_gpu
from jTransUP.utils.loss import bprLoss, orthogonalLoss, normLoss
from jTransUP.utils.visuliazer import Visualizer

FLAGS = gflags.FLAGS

def evaluate(FLAGS, model, user_total, item_total, eval_total, eval_iter, testDict, logger, num_processes=multiprocessing.cpu_count(), show_sample=False):
    # Evaluate
    total_batches = math.ceil(float(eval_total) / FLAGS.batch_size)
    # processing bar
    pbar = tqdm(total=total_batches)
    pbar.set_description("Run Eval")

    # key is u_id, value is a sorted list of size FLAGS.topn
    pred_dict = {}

    model_target = 1 if FLAGS.model_type == "bprmf" else -1

    model.eval()

    with multiprocessing.Manager() as manager:
        d = manager.dict()

        lock = multiprocessing.Lock()
        queue = multiprocessing.JoinableQueue()
        workerList = []
        for i in range(num_processes):
            worker = MyTopNRecEvalProcess(d, lock, topn=FLAGS.topn, target=model_target, queue=queue)
            workerList.append(worker)
            worker.daemon = True
            worker.start()

        for rating_batch in eval_iter:
            if rating_batch is None : break
            u_ids, i_ids = rating_batch
            score = model(u_ids, i_ids)
            queue.put((u_ids, i_ids, score.data.cpu().numpy()))
            pbar.update(1)

        queue.join()

        pred_dict = dict(d)

        for worker in workerList:
            worker.terminate()

    f1, hit, ndcg, p, r = getPerformance(pred_dict, testDict)
    pbar.close()

    logger.info("f1:{:.4f}, hit:{:.4f}, ndcg:{:.4f}, p:{:.4f}, r:{:.4f}, topn:{}.".format(f1, hit, ndcg, p, r, FLAGS.topn))

    return f1, hit, ndcg, p, r

def train_loop(FLAGS, model, trainer, train_iter, eval_iter, valid_iter,
            user_total, item_total, eval_total, validDict, testDict, logger, vis=None, num_processes=multiprocessing.cpu_count(), show_sample=False):

    # Train.
    logger.info("Training.")

    # New Training Loop
    pbar = tqdm(total=FLAGS.eval_interval_steps)
    pbar.set_description("Training")

    for _ in range(trainer.step, FLAGS.training_steps):
        total_loss = 0.0
        if (trainer.step - trainer.best_step) > FLAGS.early_stopping_steps_to_wait:
            logger.info('No improvement after ' +
                       str(FLAGS.early_stopping_steps_to_wait) +
                       ' steps. Stopping training.')
            break

        # set model in training mode
        model.train()
        
        rating_batch = next(train_iter)
        u, pi, ni = rating_batch

        model_target = 1 if FLAGS.model_type == "bprmf" else -1

        trainer.optimizer_zero_grad()

        # Run model. output: batch_size * cand_num
        pos_score = model(u, pi)
        neg_score = model(u, ni)

        # Calculate loss.
        if FLAGS.loss_type == "margin":
            losses = nn.MarginRankingLoss(margin=FLAGS.margin).forward(pos_score, neg_score, model_target)
        elif FLAGS.loss_type == "bpr":
            losses = bprLoss(pos_score, neg_score, target=model_target)
        
        if FLAGS.model_type == "transup":
            losses += orthogonalLoss(model.pref_weight, model.norm_weight)
        # Backward pass.
        losses.backward()
        # Hard Gradient Clipping
        nn.utils.clip_grad_norm([param for name, param in model.named_parameters() if name not in ["user_embeddings.weight", "item_embeddings.weight"]], FLAGS.clipping_max_value)

        # Gradient descent step.
        trainer.optimizer_step()
        total_loss += losses.data[0]

        pbar.update(1)

        if trainer.step > 0 and trainer.step % FLAGS.eval_interval_steps == 0:
            pbar.close()
            # use valid loss to record best performance
            dev_rating_batch = next(valid_iter)
            dev_u, dev_pi, dev_ni = dev_rating_batch

            # Run model. output: batch_size * cand_num
            dev_pos_score = model(dev_u, dev_pi)
            dev_neg_score = model(dev_u, dev_ni)

            # Calculate loss.
            if FLAGS.loss_type == "margin":
                dev_losses = nn.MarginRankingLoss(margin=FLAGS.margin).forward(dev_pos_score, dev_neg_score, model_target)
            elif FLAGS.loss_type == "bpr":
                dev_losses = bprLoss(dev_pos_score, dev_neg_score, target=model_target)

            if FLAGS.model_type == "transup":
                dev_losses += orthogonalLoss(model.pref_weight, model.norm_weight)

            test_performance = evaluate(FLAGS, model, user_total, item_total, eval_total, eval_iter, testDict, logger, num_processes=num_processes, show_sample=show_sample)

            trainer.new_performance(dev_losses.data[0], test_performance)
            logger.info("train loss:{:.4f}, valid loss:{:.4f}!".format(total_loss, dev_losses.data[0]))

            pbar = tqdm(total=FLAGS.eval_interval_steps)
            pbar.set_description("Training")
            # visuliazation
            if vis is not None:
                vis.plot_many_stack({'train_loss': total_loss, 'valid_loss':dev_losses.data[0],},
                win_name="Loss Curve")
                vis.plot_many_stack({'f1':test_performance[0]}, win_name="F1 Score@{}".format(FLAGS.topn))
                vis.plot_many_stack({'hit':test_performance[1]}, win_name="Hit Ratio@{}".format(FLAGS.topn))
                vis.plot_many_stack({'ndcg':test_performance[2]}, win_name="NDCG@{}".format(FLAGS.topn))
                vis.plot_many_stack({'precision':test_performance[3]}, win_name="Precision@{}".format(FLAGS.topn))
                vis.plot_many_stack({'recall':test_performance[4]}, win_name="Recall@{}".format(FLAGS.topn))

def run(only_forward=False):
    # set visualization
    vis = None
    if FLAGS.has_visualization:
        vis = Visualizer(env=FLAGS.experiment_name)

    # set logger
    logger = logging.getLogger()
    log_level = logging.DEBUG if FLAGS.log_level == "debug" else logging.INFO
    logger.setLevel(level=log_level)
    
    log_path = FLAGS.log_path + FLAGS.dataset
    if not os.path.exists(log_path) : os.makedirs(log_path)
    log_file = os.path.join(log_path, FLAGS.experiment_name + ".log")
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # FileHandler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Flag Values:\n" + json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))

    # load data
    train_iter, eval_iter, valid_iter, user_total, item_total, trainTotal, testTotal, validTotal, testDict, validDict = load_data(FLAGS, logger)

    rating_total = trainTotal + testTotal + validTotal
    eval_total = user_total*item_total - trainTotal - validTotal

    model = init_model(FLAGS, user_total, item_total, logger)
    epoch_length = math.ceil( trainTotal / FLAGS.batch_size )
    trainer = ModelTrainer(model, logger, epoch_length, FLAGS)

    # Do an evaluation-only run.
    if only_forward:
        evaluate(
            FLAGS,
            model,
            user_total,
            item_total,
            eval_total,
            eval_iter,
            testDict,
            logger,
            num_processes=FLAGS.num_processes,
            show_sample=False)
    else:
        train_loop(
            FLAGS,
            model,
            trainer,
            train_iter,
            eval_iter,
            valid_iter,
            user_total,
            item_total,
            eval_total,
            validDict,
            testDict,
            logger,
            vis=vis,
            num_processes=FLAGS.num_processes,
            show_sample=False)
    

if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)
    flag_defaults(FLAGS)

    run(only_forward=FLAGS.eval_only_mode)