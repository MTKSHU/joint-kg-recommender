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

from jTransUP.models.base import get_flags, flag_defaults, init_model
from jTransUP.data.load_triple_data import load_data
from jTransUP.utils.trainer import ModelTrainer
from jTransUP.utils.misc import Accumulator, getPerformance, evalProcess
from jTransUP.utils.misc import to_gpu
from jTransUP.utils.loss import bprLoss, orthogonalLoss, normLoss
from jTransUP.utils.visuliazer import Visualizer
from jTransUP.data.load_kg_rating_data import loadR2KgMap
from jTransUP.utils.data import getTrainTripleBatch, getTripleElements
import jTransUP.utils.loss as loss

FLAGS = gflags.FLAGS

def evaluate(FLAGS, model, entity_total, relation_total, eval_total, eval_iters, evalDicts, logger, show_sample=False):
    # Evaluate
    total_batches = math.ceil(float(eval_total) / FLAGS.batch_size)
    # processing bar
    pbar = tqdm(total=total_batches)
    pbar.set_description("Run Eval")
    # key is u_id, value is a sorted list of size FLAGS.topn

    model.eval()

    head_eval_iter, tail_eval_iter = eval_iters
    head_eval_dict, tail_eval_dict = evalDicts

    head_pred_dict = {}
    # head prediction evaluation
    for triple_batch in head_eval_iter:
        if triple_batch is None : break
        h, t, r = getTripleElements(triple_batch)
        h_var = to_gpu(V(torch.LongTensor(h)))
        t_var = to_gpu(V(torch.LongTensor(t)))
        r_var = to_gpu(V(torch.LongTensor(r)))
        score = model(h_var, t_var, r_var)
        pred_triples = zip(list(zip(t, r)), h, score.data.cpu().numpy())
        head_pred_dict = evalProcess(list(pred_triples), head_pred_dict, is_descending=False)
        pbar.update(1)

    head_f1, head_p, head_r, head_hit, head_ndcg, head_mean_rank = getPerformance(head_pred_dict, head_eval_dict, topn=FLAGS.topn)

    tail_pred_dict = {}
    # tail prediction evaluation
    for triple_batch in tail_eval_iter:
        if triple_batch is None : break
        h, t, r = getTripleElements(triple_batch)
        h_var = to_gpu(V(torch.LongTensor(h)))
        t_var = to_gpu(V(torch.LongTensor(t)))
        r_var = to_gpu(V(torch.LongTensor(r)))
        score = model(h_var, t_var, r_var)
        pred_triples = zip(list(zip(h, r)), t, score.data.cpu().numpy())
        tail_pred_dict = evalProcess(list(pred_triples), tail_pred_dict, is_descending=False)
        pbar.update(1)

    tail_f1, tail_p, tail_r, tail_hit, tail_ndcg, tail_mean_rank = getPerformance(tail_pred_dict, tail_eval_dict, topn=FLAGS.topn)

    pbar.close()

    head_num = len(head_pred_dict)
    tail_num = len(tail_pred_dict)

    avg_f1 = float(head_f1 * head_num + tail_f1 * tail_num) / (head_num + tail_num)
    avg_p = float(head_p * head_num + tail_p * tail_num) / (head_num + tail_num)
    avg_r = float(head_r * head_num + tail_r * tail_num) / (head_num + tail_num)
    avg_hit = float(head_hit * head_num + tail_hit * tail_num) / (head_num + tail_num)
    avg_ndcg = float(head_ndcg * head_num + tail_ndcg * tail_num) / (head_num + tail_num)
    avg_mean_rank = float(head_mean_rank * head_num + tail_mean_rank * tail_num) / (head_num + tail_num)

    logger.info("f1:{:.4f}, p:{:.4f}, r:{:.4f}, hit:{:.4f}, ndcg:{:.4f}, meanRank:{:.4f}, topn:{}.".format(avg_f1, avg_p, avg_r, avg_hit, avg_ndcg, avg_mean_rank, FLAGS.topn))

    return avg_f1, avg_p, avg_r, avg_hit, avg_ndcg, avg_mean_rank

def train_loop(FLAGS, model, trainer, triple_iters, datasets,
            entity_total, relation_total, actual_test_total, actual_valid_total, logger, vis=None, show_sample=False):
    train_list, test_list, valid_list, head_test_dict, tail_test_dict, head_valid_dict, tail_valid_dict, allHeadDict, allTailDict = datasets

    train_iter, test_head_iter, test_tail_iter, valid_head_iter, valid_tail_iter = triple_iters

    valid_total = len(valid_list)
    # Train.
    logger.info("Training.")

    # New Training Loop
    pbar = None
    total_loss = 0.0
    for _ in range(trainer.step, FLAGS.training_steps):
        
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
                dev_performance = evaluate(FLAGS, model, entity_total, relation_total, actual_valid_total, (valid_head_iter, valid_tail_iter), (head_valid_dict, tail_valid_dict), logger, show_sample=show_sample)

            test_performance = evaluate(FLAGS, model, entity_total, relation_total, actual_test_total, (test_head_iter, test_tail_iter), (head_test_dict, tail_test_dict), logger, show_sample=show_sample)

            if valid_total == 0: 
                dev_performance = test_performance

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
                vis.plot_many_stack({'Valid MeanRank':dev_performance[5], 'Test MeanRank':test_performance[5]}, win_name="MeanRank@{}".format(FLAGS.topn))
            total_loss = 0.0

        triple_batch = next(train_iter)
        ph, pt, pr, nh, nt, nr = getTrainTripleBatch(triple_batch, entity_total, allHeadDict, allTailDict)

        ph_var = to_gpu(V(torch.LongTensor(ph)))
        pt_var = to_gpu(V(torch.LongTensor(pt)))
        pr_var = to_gpu(V(torch.LongTensor(pr)))
        nh_var = to_gpu(V(torch.LongTensor(nh)))
        nt_var = to_gpu(V(torch.LongTensor(nt)))
        nr_var = to_gpu(V(torch.LongTensor(nr)))

        # set model in training mode
        model.train()

        trainer.optimizer_zero_grad()

        # Run model. output: batch_size * 1
        pos_score = model(ph_var, pt_var, pr_var)
        neg_score = model(nh_var, nt_var, nr_var)

        # Calculate loss.
        if FLAGS.loss_type == "margin":
            losses = nn.MarginRankingLoss(margin=FLAGS.margin).forward(pos_score, neg_score, -1.0)
        elif FLAGS.loss_type == "bpr":
            losses = bprLoss(pos_score, neg_score, target=-1.0)
        else:
            raise NotImplementedError

        ent_embeddings = model.ent_embeddings(torch.cat([ph_var, pt_var, nh_var, nt_var]))
        rel_embeddings = model.rel_embeddings(torch.cat([pr_var, nr_var]))
        norm_embeddings = model.norm_embeddings(torch.cat([pr_var, nr_var]))
		
        losses += loss.orthogonalLoss(rel_embeddings, norm_embeddings)

        losses = losses + loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings)
        
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
    # load relation vocab for filtering
    kg_path = os.path.join(os.path.join(FLAGS.data_path, FLAGS.dataset), 'kg')

    rel_vocab = None
    if FLAGS.filter_relation:
        rel_vocab = set()
        rel_file = os.path.join(kg_path, 'relation_filter.dat')
        with open(rel_file, 'r') as fin:
            for line in fin:
                rel_vocab.add(line.strip())
    
    datasets, triple_iters, e_map, r_map, entity_total, relation_total = load_data(kg_path, FLAGS.batch_size, filter_wrong_corrupted=FLAGS.filter_wrong_corrupted, rel_vocab=rel_vocab, logger=logger, hop=1, filter_unseen_samples=FLAGS.filter_unseen_samples, shuffle_data_split=FLAGS.shuffle_data_split, train_ratio=FLAGS.train_ratio, test_ratio=FLAGS.test_ratio)

    train_list, test_list, valid_list, allHeadDict, allTailDict = datasets
    train_iter, test_head_iter, test_tail_iter, valid_head_iter, valid_tail_iter = triple_iters

    trainTotal = len(train_list)
    validTotal = len(valid_list)
    testTotal = len(test_list)

    # get the number of (h,r) and (r,t) in test and valid
    head_test_dict = {}
    tail_test_dict = {}
    for triple in test_list:
        tmp_set = head_test_dict.get( (triple.t, triple.r), set())
        tmp_set.add(triple.h)
        head_test_dict[(triple.t, triple.r)] = tmp_set
        
        tmp_set = tail_test_dict.get( (triple.h, triple.r), set())
        tmp_set.add(triple.t)
        tail_test_dict[(triple.h, triple.r)] = tmp_set

    num_test = len(head_test_dict) + len(tail_test_dict)

    head_valid_dict = {}
    tail_valid_dict = {}
    for triple in valid_list:
        tmp_set = head_valid_dict.get( (triple.t, triple.r), set())
        tmp_set.add(triple.h)
        head_valid_dict[(triple.t, triple.r)] = tmp_set
        
        tmp_set = tail_valid_dict.get( (triple.h, triple.r), set())
        tmp_set.add(triple.t)
        tail_valid_dict[(triple.h, triple.r)] = tmp_set

    num_valid = len(head_valid_dict) + len(tail_valid_dict)

    actual_test_total = num_test * entity_total - 2*trainTotal - 2*validTotal
    actual_valid_total = 0 if validTotal > 0 is None else num_valid * entity_total - 2 * trainTotal - 2 * testTotal

    datasets = train_list, test_list, valid_list, head_test_dict, tail_test_dict, head_valid_dict, tail_valid_dict, allHeadDict, allTailDict

    model = init_model(FLAGS, entity_total, relation_total, logger)
    epoch_length = math.ceil( trainTotal / FLAGS.batch_size )
    trainer = ModelTrainer(model, logger, epoch_length, FLAGS)

    # Do an evaluation-only run.
    if only_forward:
        evaluate(
            FLAGS,
            model,
            entity_total,
            relation_total,
            actual_test_total,
            (test_head_iter, test_tail_iter),
            (head_test_dict, tail_test_dict),
            logger,
            show_sample=False)
    else:
        train_loop(
            FLAGS,
            model,
            trainer,
            triple_iters,
            datasets,
            entity_total,
            relation_total,
            actual_test_total,
            actual_valid_total,
            logger,
            vis=vis,
            show_sample=False)

if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)
    flag_defaults(FLAGS)

    run(only_forward=FLAGS.eval_only_mode)