import logging
import gflags
import sys
import os
import json
from tqdm import tqdm
tqdm.monitor_iterval=0
import math
import time
import random

import torch
import torch.nn as nn
from torch.autograd import Variable as V

from jTransUP.models.base import get_flags, flag_defaults, load_data
from jTransUP.models.base import init_model
from jTransUP.utils.trainer import ModelTrainer
from jTransUP.utils.misc import Accumulator, getPerformance, evalProcess
from jTransUP.utils.misc import to_gpu
from jTransUP.utils.loss import bprLoss, orthogonalLoss, normLoss
from jTransUP.utils.visuliazer import Visualizer

FLAGS = gflags.FLAGS

def evaluate(FLAGS, model, user_total, item_total, eval_total, eval_iter, evalDict, logger, show_sample=False):
    # Evaluate
    total_batches = math.ceil(float(eval_total) / FLAGS.batch_size)
    # processing bar
    pbar = tqdm(total=total_batches)
    pbar.set_description("Run Eval")

    # key is u_id, value is a sorted list of size FLAGS.topn
    pred_dict = {}

    model_target = 1 if FLAGS.model_type == "bprmf" else -1

    model.eval()

    pred_dict = {}
    for rating_batch in eval_iter:
        if rating_batch is None : break
        u_ids, i_ids = rating_batch
        score = model(u_ids, i_ids)
        pred_ratings = zip(u_ids, i_ids, score.data.cpu().numpy())
        pred_dict = evalProcess(list(pred_ratings), pred_dict, topn=FLAGS.topn, target=model_target)
        pbar.update(1)
        
    f1, p, r, hit, ndcg = getPerformance(pred_dict, evalDict)
    pbar.close()

    logger.info("f1:{:.4f}, p:{:.4f}, r:{:.4f}, hit:{:.4f}, ndcg:{:.4f}, topn:{}.".format(f1, p, r, hit, ndcg, FLAGS.topn))

    return f1, p, r, hit, ndcg

def train_loop(FLAGS, model, trainer, train_iter, test_iter, valid_iter,
            user_total, item_total, test_total, valid_total, validDict, testDict, logger, vis=None, show_sample=False):

    # Train.
    logger.info("Training.")

    # New Training Loop
    pbar = None
    total_loss = 0.0
    for _ in range(trainer.step, FLAGS.training_steps):
        
        model_target = 1 if FLAGS.model_type == "bprmf" else -1

        if FLAGS.early_stopping_steps_to_wait > 0 and (trainer.step - trainer.best_step) > FLAGS.early_stopping_steps_to_wait:
            logger.info('No improvement after ' +
                       str(FLAGS.early_stopping_steps_to_wait) +
                       ' steps. Stopping training.')
            if pbar is not None: pbar.close()
            break
        if trainer.step % FLAGS.eval_interval_steps == 0:
            if pbar is not None:
                pbar.close()
            total_loss /= FLAGS.eval_interval_steps
            logger.info("train loss:{:.4f}!".format(total_loss))
            
            if valid_total > 0:
                dev_performance = evaluate(FLAGS, model, user_total, item_total, valid_total, valid_iter, validDict, logger, show_sample=show_sample)
            
            test_performance = evaluate(FLAGS, model, user_total, item_total, test_total, test_iter, testDict, logger, show_sample=show_sample)
            if valid_total == 0: dev_performance = test_performance
            trainer.new_performance(dev_performance, test_performance)

            pbar = tqdm(total=FLAGS.eval_interval_steps)
            pbar.set_description("Training")
            # visuliazation
            if vis is not None:
                vis.plot_many_stack({'Train Loss': total_loss},
                win_name="Loss Curve")
                vis.plot_many_stack({'Valid F1':dev_performance[0], 'Test F1':test_performance[0]}, win_name="F1 Score@{}".format(FLAGS.topn))
                vis.plot_many_stack({'Valid Precision':dev_performance[1], 'Test Precision':test_performance[1]}, win_name="Precision@{}".format(FLAGS.topn))
                vis.plot_many_stack({'Valid Recall':dev_performance[2], 'Test Recall':test_performance[2]}, win_name="Recall@{}".format(FLAGS.topn))
                vis.plot_many_stack({'Valid Hit':dev_performance[3], 'Test Hit':test_performance[3]}, win_name="Hit Ratio@{}".format(FLAGS.topn))
                vis.plot_many_stack({'Valid NDCG':dev_performance[4], 'Test NDCG':test_performance[4]}, win_name="NDCG@{}".format(FLAGS.topn))
            total_loss = 0.0

        rating_batch = next(train_iter)
        u, pi, ni = rating_batch

        # set model in training mode
        model.train()

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

        # for param in model.parameters():
        #     print(param.grad.data.sum())

        # Hard Gradient Clipping
        nn.utils.clip_grad_norm([param for name, param in model.named_parameters()], FLAGS.clipping_max_value)

        # Gradient descent step.
        trainer.optimizer_step()
        total_loss += losses.data[0]
        pbar.update(1)

def run(only_forward=False):
    if FLAGS.seed != 0:
        random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)

    # set visualization
    vis = None
    if FLAGS.has_visualization:
        vis = Visualizer(env=FLAGS.experiment_name)
        vis.log(json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True),
                win_name="Parameter")

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
    rating_datasets, triple_datasets, i2kg_map, kg2i_map = load_data(FLAGS, logger)

    # construct data iter

    test_total = user_total*item_total - trainTotal - validTotal
    valid_total = 0 if valid_iter is None else user_total*item_total - trainTotal - testTotal

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
            test_total,
            test_iter,
            testDict,
            logger,
            show_sample=False)
    else:
        train_loop(
            FLAGS,
            model,
            trainer,
            train_iter,
            test_iter,
            valid_iter,
            user_total,
            item_total,
            test_total,
            valid_total,
            validDict,
            testDict,
            logger,
            vis=vis,
            show_sample=False)
    

if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)
    flag_defaults(FLAGS)

    run(only_forward=FLAGS.eval_only_mode)