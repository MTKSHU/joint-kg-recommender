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
from jTransUP.data.load_kg_rating_data import load_data
from jTransUP.utils.trainer import ModelTrainer
from jTransUP.utils.misc import Accumulator, getPerformance, evalProcess
from jTransUP.utils.misc import to_gpu
from jTransUP.utils.loss import bprLoss, orthogonalLoss, normLoss
from jTransUP.utils.visuliazer import Visualizer
from jTransUP.data.load_kg_rating_data import loadR2KgMap
from jTransUP.utils.data import getTrainTripleBatch, getTripleElements, getTrainRatingBatch

FLAGS = gflags.FLAGS

def getMappedEntities(i_ids, i_remap, new_map):
    e_ids = []
    new_i_ids = []
    for i in set(i_ids):
        new_index = new_map[i_remap[i]]
        if new_index[0] == -1: continue
        e_ids.append(new_index[0])
        new_i_ids.append(new_index[1])
    return e_ids, new_i_ids

def getMappedItems(e_ids, e_remap, new_map):
    i_ids = []
    new_e_ids = []
    for e in set(e_ids):
        new_index = new_map[e_remap[e]]
        if new_index[1] == -1: continue
        i_ids.append(new_index[1])
        new_e_ids.append(new_index[0])
    return new_e_ids, i_ids

def evaluateRec(FLAGS, model, user_total, item_total, eval_total, eval_iter, evalDict, logger, show_sample=False):
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
        u_ids, i_ids = getEvalRatingBatch(rating_batch)
        u_var = to_gpu(V(torch.LongTensor(u_ids)))
        i_var = to_gpu(V(torch.LongTensor(i_ids)))
        score = model( (u_var, i_var), None, is_rec=True)
        pred_ratings = zip(u_ids, i_ids, score.data.cpu().numpy())
        pred_dict = evalProcess(list(pred_ratings), pred_dict, is_descending=True if model_target==1 else False)
        pbar.update(1)
    pbar.close()
    
    f1, p, r, hit, ndcg, mean_rank = getPerformance(pred_dict, evalDict, topn=FLAGS.topn)

    logger.info("f1:{:.4f}, p:{:.4f}, r:{:.4f}, hit:{:.4f}, ndcg:{:.4f}, mean_rank:{:.4f}, topn:{}.".format(f1, p, r, hit, ndcg, mean_rank, FLAGS.topn))

    return f1, p, r, hit, ndcg, mean_rank

def evaluateKG(FLAGS, model, entity_total, relation_total, eval_total, eval_iters, evalDicts, logger, show_sample=False):
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
        score = model(None, (h_var, t_var, r_var), is_rec=False)
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
        score = model(None, (h_var, t_var, r_var), is_rec=False)
        pred_triples = zip(list(zip(h, r)), t, score.data.cpu().numpy())
        tail_pred_dict = evalProcess(list(pred_triples), tail_pred_dict, is_descending=False)
        pbar.update(1)

    tail_f1, tail_p, tail_r, tail_hit, tail_ndcg, tail_mean_rank = getPerformance(tail_pred_dict, tail_eval_dict, topn=FLAGS.topn)

    pbar.close()

    logger.info("hf1:{:.4f}, hp:{:.4f}, hr:{:.4f}, hhit:{:.4f}, hndcg:{:.4f}, hmeanRank:{:.4f}, topn:{}.".format(head_f1, head_p, head_r, head_hit, head_ndcg, head_mean_rank, FLAGS.topn))

    logger.info("tf1:{:.4f}, tp:{:.4f}, tr:{:.4f}, thit:{:.4f}, tndcg:{:.4f}, tmeanRank:{:.4f}, topn:{}.".format(tail_f1, tail_p, tail_r, tail_hit, tail_ndcg, tail_mean_rank, FLAGS.topn))

    head_num = len(head_pred_dict)
    tail_num = len(tail_pred_dict)

    avg_f1 = float(head_f1 * head_num + tail_f1 * tail_num) / (head_num + tail_num)
    avg_p = float(head_p * head_num + tail_p * tail_num) / (head_num + tail_num)
    avg_r = float(head_r * head_num + tail_r * tail_num) / (head_num + tail_num)
    avg_hit = float(head_hit * head_num + tail_hit * tail_num) / (head_num + tail_num)
    avg_ndcg = float(head_ndcg * head_num + tail_ndcg * tail_num) / (head_num + tail_num)
    avg_mean_rank = float(head_mean_rank * head_num + tail_mean_rank * tail_num) / (head_num + tail_num)

    logger.info("af1:{:.4f}, ap:{:.4f}, ar:{:.4f}, ahit:{:.4f}, andcg:{:.4f}, ameanRank:{:.4f}, topn:{}.".format(avg_f1, avg_p, avg_r, avg_hit, avg_ndcg, avg_mean_rank, FLAGS.topn))

    return avg_f1, avg_p, avg_r, avg_hit, avg_ndcg, avg_mean_rank

def train_loop(FLAGS, model, trainer, rating_iters, triple_iters, rating_datasets, triple_datasets, statistics, new_map, e_remap, i_remap, logger, vis=None, show_sample=False):
    rating_train_list, rating_test_dict, rating_valid_dict, rating_all_dict, rating_test_total, rating_valid_total = rating_datasets

    triple_train_list, triple_test_list, triple_valid_list, triple_test_Hdict, triple_test_Tdict, triple_valid_Hdict, triple_valid_Tdict, triple_all_Hdict, triple_all_Tdict = triple_datasets

    rating_train_iter, rating_test_iter, rating_valid_iter = rating_iters

    triple_train_iter, triple_test_Hiter, triple_test_Titer, triple_valid_Hiter, triple_valid_Titer = triple_iters

    user_total, item_total, entity_total, relation_total, actual_rating_test_total, actual_rating_valid_total, actual_triple_test_total, actual_triple_valid_total = statistics

    triple_valid_total = len(triple_valid_list)

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

            if rating_valid_total > 0:
                rating_dev_performance = evaluateRec(FLAGS, model, user_total, item_total, actual_rating_valid_total, rating_valid_iter, rating_valid_dict, logger, show_sample=show_sample)
            if triple_valid_total > 0:
                kg_dev_performance = evaluateKG(FLAGS, model, entity_total, relation_total, actual_triple_valid_total, (triple_valid_Hiter, triple_valid_Titer), (triple_valid_Hdict, triple_valid_Tdict), logger, show_sample=show_sample)

            rating_test_performance = evaluateRec(FLAGS, model, user_total, item_total, actual_rating_test_total, rating_test_iter, rating_test_dict, logger, show_sample=show_sample)

            kg_test_performance = evaluateKG(FLAGS, model, entity_total, relation_total, actual_triple_test_total, (triple_test_Hiter, triple_test_Titer), (triple_test_Hdict, triple_test_Tdict), logger, show_sample=show_sample)

            if rating_valid_total == 0: 
                rating_dev_performance = rating_test_performance
            if triple_valid_total == 0:
                kg_dev_performance = kg_test_performance

            trainer.new_performance(rating_dev_performance, rating_test_performance)

            pbar = tqdm(total=FLAGS.eval_interval_steps)
            pbar.set_description("Training")
            # visuliazation
            if vis is not None:
                vis.plot_many_stack({'Train Loss': total_loss},
                win_name="Loss Curve")
                # recommendation performance
                vis.plot_many_stack({'Valid F1':rating_dev_performance[0], 'Test F1':rating_test_performance[0]}, win_name="Recommendation F1 Score@{}".format(FLAGS.topn))

                vis.plot_many_stack({'Valid Precision':rating_dev_performance[1], 'Test Precision':rating_test_performance[1]}, win_name="Recommendation Precision@{}".format(FLAGS.topn))
                vis.plot_many_stack({'Valid Recall':rating_dev_performance[2], 'Test Recall':rating_test_performance[2]}, win_name="Recommendation Recall@{}".format(FLAGS.topn))
                vis.plot_many_stack({'Valid Hit':rating_dev_performance[3], 'Test Hit':rating_test_performance[3]}, win_name="Recommendation Hit Ratio@{}".format(FLAGS.topn))
                vis.plot_many_stack({'Valid NDCG':rating_dev_performance[4], 'Test NDCG':rating_test_performance[4]}, win_name="Recommendation NDCG@{}".format(FLAGS.topn))
                vis.plot_many_stack({'Valid MeanRank':rating_dev_performance[5], 'Test MeanRank':rating_test_performance[5]}, win_name="Recommendation MeanRank@{}".format(FLAGS.topn))

                # KG performance
                vis.plot_many_stack({'Valid F1':kg_dev_performance[0], 'Test F1':kg_test_performance[0]}, win_name="Link Prediction F1 Score@{}".format(FLAGS.topn))

                vis.plot_many_stack({'Valid Precision':kg_dev_performance[1], 'Test Precision':kg_test_performance[1]}, win_name="Link Prediction Precision@{}".format(FLAGS.topn))
                vis.plot_many_stack({'Valid Recall':kg_dev_performance[2], 'Test Recall':kg_test_performance[2]}, win_name="Link Prediction Recall@{}".format(FLAGS.topn))
                vis.plot_many_stack({'Valid Hit':kg_dev_performance[3], 'Test Hit':kg_test_performance[3]}, win_name="Link Prediction Hit Ratio@{}".format(FLAGS.topn))
                vis.plot_many_stack({'Valid NDCG':kg_dev_performance[4], 'Test NDCG':kg_test_performance[4]}, win_name="Link Prediction NDCG@{}".format(FLAGS.topn))
                vis.plot_many_stack({'Valid MeanRank':kg_dev_performance[5], 'Test MeanRank':kg_test_performance[5]}, win_name="Link Prediction MeanRank@{}".format(FLAGS.topn))
            total_loss = 0.0

        rating_batch = next(rating_train_iter)
        triple_batch = next(triple_train_iter)

        u, pi, ni = getTrainRatingBatch(rating_batch, item_total, rating_all_dict)
        ph, pt, pr, nh, nt, nr = getTrainTripleBatch(triple_batch, entity_total, triple_all_Hdict, triple_all_Tdict)

        u_var = to_gpu(V(torch.LongTensor(u)))
        pi_var = to_gpu(V(torch.LongTensor(pi)))
        ni_var = to_gpu(V(torch.LongTensor(ni)))

        ph_var = to_gpu(V(torch.LongTensor(ph)))
        pt_var = to_gpu(V(torch.LongTensor(pt)))
        pr_var = to_gpu(V(torch.LongTensor(pr)))
        nh_var = to_gpu(V(torch.LongTensor(nh)))
        nt_var = to_gpu(V(torch.LongTensor(nt)))
        nr_var = to_gpu(V(torch.LongTensor(nr)))

        # set model in training mode
        model.train()

        trainer.optimizer_zero_grad()

        # Run model. recommendation model
        rating_pos_score = model( (u_var, pi_var), None, is_rec=True)
        rating_neg_score = model( (u_var, ni_var), None, is_rec=True)

        kg_pos_score = model( None, (ph_var, pt_var, pr_var), is_rec=False)
        kg_neg_score = model( None, (nh_var, nt_var, nr_var), is_rec=False)

        # Calculate loss.
        losses = bprLoss(rating_pos_score, rating_neg_score, target=-1.0)

        losses += nn.MarginRankingLoss(margin=FLAGS.margin).forward(kg_pos_score, kg_neg_score, to_gpu(torch.autograd.Variable(torch.FloatTensor([-1.0]*len(ph)))))

        ent_embeddings = model.ent_embeddings(torch.cat([ph_var, pt_var, nh_var, nt_var]))
        rel_embeddings = model.rel_embeddings(torch.cat([pr_var, nr_var]))
        norm_embeddings = model.norm_embeddings(torch.cat([pr_var, nr_var]))
		
        losses += orthogonalLoss(rel_embeddings, norm_embeddings)

        losses = losses + normLoss(ent_embeddings) + normLoss(rel_embeddings)
        
        # shared loss
        if not FLAGS.share_embeddings:
            e_ids1, i_ids1 = getMappedEntities(pi+ni, i_remap, new_map)
            e_ids2, i_ids2 = getMappedItems(ph+pt+nh+nt, e_remap, new_map)
            e_var = to_gpu(V(torch.LongTensor(e_ids1+e_ids2)))
            i_var = to_gpu(V(torch.LongTensor(i_ids1+i_ids2)))
            ent_embeddings = model.ent_embeddings(e_var)
            item_embeddings = model.item_embeddings(i_var)
            losses += nn.MSELoss()(ent_embeddings, item_embeddings)

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
    rating_datasets, rating_iters, u_map, i_map, user_total, item_total, triple_datasets, triple_iters, e_map, r_map, entity_total, relation_total, new_map, e_remap, i_remap = load_data(FLAGS, logger=logger, hop=1)

    # organize triple data
    triple_train_list, triple_test_list, triple_valid_list, triple_all_Hdict, triple_all_Tdict = triple_datasets

    triple_train_iter, triple_test_Hiter, triple_test_Titer, triple_valid_Hiter, triple_valid_Titer = triple_iters

    triple_train_total = len(triple_train_list)
    triple_valid_total = len(triple_valid_list)
    triple_test_total = len(triple_test_list)

    # get the number of (h,r) and (r,t) in test and valid
    triple_test_Hdict = {}
    triple_test_Tdict = {}
    for triple in triple_test_list:
        tmp_set = triple_test_Hdict.get( (triple.t, triple.r), set())
        tmp_set.add(triple.h)
        triple_test_Hdict[(triple.t, triple.r)] = tmp_set
        
        tmp_set = triple_test_Tdict.get( (triple.h, triple.r), set())
        tmp_set.add(triple.t)
        triple_test_Tdict[(triple.h, triple.r)] = tmp_set

    triple_test_num = len(triple_test_Hdict) + len(triple_test_Tdict)

    triple_valid_Hdict = {}
    triple_valid_Tdict = {}
    for triple in triple_valid_list:
        tmp_set = triple_valid_Hdict.get( (triple.t, triple.r), set())
        tmp_set.add(triple.h)
        triple_valid_Hdict[(triple.t, triple.r)] = tmp_set
        
        tmp_set = triple_valid_Tdict.get( (triple.h, triple.r), set())
        tmp_set.add(triple.t)
        triple_valid_Tdict[(triple.h, triple.r)] = tmp_set

    triple_valid_num = len(triple_valid_Hdict) + len(triple_valid_Tdict)

    actual_triple_test_total = triple_test_num * entity_total - 2*triple_train_total - 2*triple_valid_total
    actual_triple_valid_total = 0 if triple_valid_total > 0 else triple_valid_num * entity_total - 2 * triple_train_total - 2 * triple_test_total

    triple_datasets = (triple_train_list, triple_test_list, triple_valid_list, triple_test_Hdict, triple_test_Tdict, triple_valid_Hdict, triple_valid_Tdict, triple_all_Hdict, triple_all_Tdict)

    # organize rating data
    rating_train_list, rating_test_dict, rating_valid_dict, rating_all_dict, rating_test_total, rating_valid_total = rating_datasets

    rating_train_iter, rating_test_iter, rating_valid_iter = rating_iters
    
    rating_train_total = len(rating_train_list)

    actual_rating_test_total = len(rating_test_dict) * item_total - rating_train_total - rating_valid_total

    actual_rating_valid_total = 0 if rating_valid_iter is None else len(rating_valid_dict) * item_total - rating_train_total - rating_test_total

    statistics = (user_total, item_total, entity_total, relation_total, actual_rating_test_total, actual_rating_valid_total, actual_triple_test_total, actual_triple_valid_total)

    if FLAGS.share_embeddings:
        item_entity_total = len(new_map)
        joint_model = init_model(FLAGS, user_total, item_entity_total, item_entity_total, relation_total, logger)
    else:
        joint_model = init_model(FLAGS, user_total, item_total, entity_total, relation_total, logger)

    epoch_length = math.ceil( (triple_train_total+rating_train_total) / FLAGS.batch_size )
    
    trainer = ModelTrainer(joint_model, logger, epoch_length, FLAGS)

    # Do an evaluation-only run.
    if only_forward:
        evaluateRec(
            FLAGS,
            joint_model,
            user_total,
            item_total,
            actual_rating_test_total,
            rating_test_iter,
            rating_test_dict,
            logger,
            show_sample=False)
        evaluateKG(
            FLAGS,
            joint_model,
            entity_total,
            relation_total,
            actual_triple_test_total,
            (triple_test_Hiter, triple_test_Titer),
            (triple_test_Hdict, triple_test_Tdict),
            logger,
            show_sample=False)
    else:
        train_loop(
            FLAGS,
            joint_model,
            trainer,
            rating_iters,
            triple_iters,
            rating_datasets,
            triple_datasets,
            statistics,
            new_map,
            e_remap,
            i_remap,
            logger,
            vis=vis,
            show_sample=False)

if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)
    flag_defaults(FLAGS)

    run(only_forward=FLAGS.eval_only_mode)