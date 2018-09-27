import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from data import *
from evaluation import *
import model
import loss

import os

from hyperboard import Agent

USE_CUDA = torch.cuda.is_available()
DATASET_PATH = './datasets/'

"""
The meaning of parameters:
self.dataset: Which dataset is used to train the model? Such as 'FB15k', 'WN18', etc.
self.learning_rate: Initial learning rate (lr) of the model.
self.early_stopping_round: How many times will lr decrease? If set to 0, it remains constant.
self.L1_flag: If set to True, use L1 distance as dissimilarity; else, use L2.
self.embedding_size: The embedding size of entities and relations.
self.train_times: The maximum number of epochs for training.
self.margin: The margin set for MarginLoss.
self.filter: Whether to check a generated negative sample is false negative.
self.momentum: The momentum of the optimizer.
self.optimizer: Which optimizer to use? Such as SGD, Adam, etc.
self.loss_function: Which loss function to use? Typically, we use margin loss.
self.entity_total: The number of different entities.
self.relation_total: The number of different relations.
self.batch_size: How many instances is contained in one batch?
self.preference_total: how many preferences shared by users?
"""

class Config(object):
	def __init__(self):
		self.dataset = None
		self.learning_rate = 0.001
		self.early_stopping_round = 0
		self.L1_flag = True
		self.embedding_size = 100
		self.train_times = 1000
		self.margin = 1.0
		self.filter = True
		self.momentum = 0.9
		self.optimizer = optim.Adam
		self.loss_function = loss.marginLoss
		self.user_total = 0
		self.item_total = 0
		self.batch_size = 32
		self.preference_total = 0
		self.neg_test_samples = 100
		self.topn = 10

if __name__ == "__main__":

	import argparse
	argparser = argparse.ArgumentParser()

	"""
	The meaning of some parameters:
	seed: Fix the random seed. Except for 0, which means no setting of random seed.
	port: The port number used by hyperboard, 
	which is a demo showing training curves in real time.
	num_processes: Number of processes used to evaluate the result.
	"""

	argparser.add_argument('-d', '--dataset', type=str)
	argparser.add_argument('-l', '--learning_rate', type=float, default=0.001)
	argparser.add_argument('-es', '--early_stopping_round', type=int, default=5)
	argparser.add_argument('-L', '--L1_flag', type=int, default=0)
	argparser.add_argument('-em', '--embedding_size', type=int, default=64)
	argparser.add_argument('-b', '--batch_size', type=int, default=512)
	argparser.add_argument('-n', '--train_times', type=int, default=100)
	argparser.add_argument('-m', '--margin', type=float, default=1.0)
	argparser.add_argument('-f', '--filter', type=int, default=1)
	argparser.add_argument('-mo', '--momentum', type=float, default=0.9)
	argparser.add_argument('-s', '--seed', type=int, default=0)
	argparser.add_argument('-op', '--optimizer', type=int, default=1)
	argparser.add_argument('-lo', '--loss_type', type=int, default=0)
	argparser.add_argument('-p', '--port', type=int, default=5000)
	argparser.add_argument('-np', '--num_processes', type=int, default=4)
	argparser.add_argument('-rp', '--num_preferences', type=int, default=6)
	argparser.add_argument('-ng', '--neg_test_samples', type=int, default=100)
	argparser.add_argument('-tn', '--topn', type=int, default=10)

	args = argparser.parse_args()

	# Start the hyperboard agent
	agent = Agent(address='127.0.0.1', port=args.port)
	if args.seed != 0:
		torch.manual_seed(args.seed)
	
	trainTotal, trainList, trainDict = loadRatings(DATASET_PATH + args.dataset, 'train.dat')
	testTotal, testList, testDict = loadRatings(DATASET_PATH + args.dataset, 'test.dat')
	allRatingDict = deepcopy(trainDict)
	for u in testDict:
		tmp_item_set = allRatingDict.get(u, set())
		tmp_item_set |= testDict[u]
		allRatingDict[u] = tmp_item_set
    # configuration for model
	config = Config()
	config.dataset = args.dataset
	config.learning_rate = args.learning_rate

	config.early_stopping_round = args.early_stopping_round

	if args.L1_flag == 1:
		config.L1_flag = True
	else:
		config.L1_flag = False

	config.embedding_size = args.embedding_size
	config.batch_size = args.batch_size
	config.train_times = args.train_times
	config.margin = args.margin
	config.neg_test_samples = args.neg_test_samples
	config.topn = args.topn

	if args.filter == 1:
		config.filter = True
	else:
		config.filter = False

	config.momentum = args.momentum

	if args.optimizer == 0:
		config.optimizer = optim.SGD
	elif args.optimizer == 1:
		config.optimizer = optim.Adam
	elif args.optimizer == 2:
		config.optimizer = optim.RMSprop

	if args.loss_type == 0:
		config.loss_function = loss.marginLoss

    # todo: preprocess user2id and item2id
	config.user_total = getAnythingTotal(DATASET_PATH + config.dataset, 'u_map.dat')
	config.item_total = getAnythingTotal(DATASET_PATH + config.dataset, 'i_map.dat')
	config.preference_total = args.num_preferences

    # parameters and curves for visulization in hyperboard
	shareHyperparameters = {'dataset': args.dataset,
		'learning_rate': args.learning_rate,
		'early_stopping_round': args.early_stopping_round,
		'L1_flag': args.L1_flag,
		'embedding_size': args.embedding_size,
		'margin': args.margin,
		'filter': args.filter,
		'momentum': args.momentum,
		'seed': args.seed,
		'optimizer': args.optimizer,
		'loss_type': args.loss_type,
		'num_preferences': args.num_preferences,
		}

	trainHyperparameters = shareHyperparameters.copy()
	trainHyperparameters.update({'type': 'train_loss'})

	validHyperparameters = shareHyperparameters.copy()
	validHyperparameters.update({'type': 'valid_loss'})

	hitHyperparameters = shareHyperparameters.copy()
	hitHyperparameters.update({'type': 'hit'})

	ndcgHyperparameters = shareHyperparameters.copy()
	ndcgHyperparameters.update({'type': 'ndcg'})

	trainCurve = agent.register(trainHyperparameters, 'train loss', overwrite=True)
	validCurve = agent.register(validHyperparameters, 'valid loss', overwrite=True)
	hitCurve = agent.register(hitHyperparameters, 'hit', overwrite=True)
	ndcgCurve = agent.register(ndcgHyperparameters, 'ndcg', overwrite=True)

	loss_function = config.loss_function()
	model = model.TransUPModel(config)

	if USE_CUDA:
		model.cuda()
		loss_function.cuda()
		longTensor = torch.cuda.LongTensor
		floatTensor = torch.cuda.FloatTensor

	else:
		longTensor = torch.LongTensor
		floatTensor = torch.FloatTensor

	optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
	margin = autograd.Variable(floatTensor([config.margin]))

	start_time = time.time()

	filename = '_'.join(
		['l', str(args.learning_rate),
		 'es', str(args.early_stopping_round),
		 'L', str(args.L1_flag),
		 'em', str(args.embedding_size),
		 'b', str(args.batch_size),
		 'n', str(args.train_times),
		 'm', str(args.margin),
		 'f', str(args.filter),
		 'mo', str(args.momentum),
		 's', str(args.seed),
		 'op', str(args.optimizer),
		 'lo', str(args.loss_type),
		 'rp', str(args.num_preferences),]) + '_TransUP.ckpt'

	trainBatchList = getBatchList(trainList, config.batch_size)

	# key := user id, value := list of negsamples
	negTestList = getTestNegList(testList, allRatingDict, config.item_total, config.neg_test_samples)
	print("ready! training ...")
	for epoch in range(config.train_times):
		total_loss = floatTensor([0.0])
		random.shuffle(trainBatchList)
		for batchList in trainBatchList:
			if config.filter == True:
				pos_u_batch, pos_i_batch, neg_u_batch, neg_i_batch = getRatingAll(batchList, 
					config.user_total, config.item_total, ratingDict=allRatingDict)
			else :
				pos_u_batch, pos_i_batch, neg_u_batch, neg_i_batch = getRatingAll(batchList, 
					config.user_total, config.item_total)

			pos_u_batch = autograd.Variable(longTensor(pos_u_batch))
			pos_i_batch = autograd.Variable(longTensor(pos_i_batch))
			neg_u_batch = autograd.Variable(longTensor(neg_u_batch))
			neg_i_batch = autograd.Variable(longTensor(neg_i_batch))

			model.zero_grad()
			pos, neg, pref, norm = model(pos_u_batch, pos_i_batch, neg_u_batch, neg_i_batch)

			if args.loss_type == 0:
				losses = loss_function(pos, neg, margin)
			else:
				losses = loss_function(pos, neg)
			
			losses += loss.orthogonalLoss(pref, norm)
			
			user_embeddings = model.user_embeddings(torch.cat([pos_u_batch, neg_u_batch]))
			item_embeddings = model.item_embeddings(torch.cat([pos_i_batch, neg_i_batch]))

			losses = losses + loss.normLoss(user_embeddings) + loss.normLoss(item_embeddings) + loss.normLoss(pref)
			
			losses.backward()
			optimizer.step()
			total_loss += losses.data

			model.norm_weight.data = F.normalize(model.norm_weight.data, p=2, dim=1)

		agent.append(trainCurve, epoch, total_loss[0])

		if epoch % 10 == 0:
			now_time = time.time()
			print(now_time - start_time)
			print("Train total loss: %d %f" % (epoch, total_loss[0]))

		if epoch % 10 == 0:
			if config.filter == True:
				pos_u_batch, pos_i_batch, neg_u_batch, neg_i_batch = getRatingBatch(batchList, config.batch_size, config.user_total, config.item_total, ratingDict=allRatingDict)
			else :
				pos_u_batch, pos_i_batch, neg_u_batch, neg_i_batch = getRatingBatch(batchList, config.batch_size, config.user_total, config.item_total)

			pos_u_batch = autograd.Variable(longTensor(pos_u_batch))
			pos_i_batch = autograd.Variable(longTensor(pos_i_batch))
			neg_u_batch = autograd.Variable(longTensor(neg_u_batch))
			neg_i_batch = autograd.Variable(longTensor(neg_i_batch))

			pos, neg, pref, norm = model(pos_u_batch, pos_i_batch, neg_u_batch, neg_i_batch)

			if args.loss_type == 0:
				losses = loss_function(pos, neg, margin)
			else:
				losses = loss_function(pos, neg)
			
			losses += loss.orthogonalLoss(pref, norm)
			
			user_embeddings = model.user_embeddings(torch.cat([pos_u_batch, neg_u_batch]))
			item_embeddings = model.item_embeddings(torch.cat([pos_i_batch, neg_i_batch]))

			losses = losses + loss.normLoss(user_embeddings) + loss.normLoss(item_embeddings) + loss.normLoss(pref)

			print("Test batch loss: %d %f" % (epoch, losses.data[0]))
			agent.append(validCurve, epoch, losses.data[0])

		if config.early_stopping_round > 0:
			if epoch == 0:
				user_embeddings = model.user_embeddings.weight.data
				item_embeddings = model.item_embeddings.weight.data
				pref_embeddings = model.pref_weight.data
				norm_embeddings = model.norm_weight.data

				L1_flag = model.L1_flag
				isfilter = model.filter
				hits, best_ndcg = evaluation_transUP(testList, negTestList, user_embeddings, item_embeddings, pref_embeddings, norm_embeddings, L1_flag, config.batch_size, config.neg_test_samples, topn=config.topn, num_processes=args.num_processes)
				print("epoch {} : hits@{} : {}, ndcg@{} : {}.".format(epoch, config.topn, hits, config.topn, best_ndcg))
				agent.append(hitCurve, epoch, hits)
				agent.append(ndcgCurve, epoch, best_ndcg)
				torch.save(model, os.path.join('./model/' + args.dataset, filename))

				best_epoch = 0
				ndcg_not_decrease_time = 0
				lr_decrease_time = 0
				#if USE_CUDA:
					#model.cuda()

			# Evaluate on validation set for every 5 epochs
			elif epoch % 5 == 0:
				user_embeddings = model.user_embeddings.weight.data
				item_embeddings = model.item_embeddings.weight.data
				pref_embeddings = model.pref_weight.data
				norm_embeddings = model.norm_weight.data

				L1_flag = model.L1_flag
				isfilter = model.filter
				hits, now_ndcg = evaluation_transUP(testList, negTestList, user_embeddings, item_embeddings, pref_embeddings, norm_embeddings, L1_flag, config.batch_size, config.neg_test_samples, topn=config.topn, num_processes=args.num_processes)

				agent.append(hitCurve, epoch, hits)
				agent.append(ndcgCurve, epoch, now_ndcg)

				if now_ndcg > best_ndcg:
					ndcg_not_decrease_time = 0
					best_ndcg = now_ndcg
					torch.save(model, os.path.join('./model/' + args.dataset, filename))
				else:
					ndcg_not_decrease_time += 1
					# If the result hasn't improved for consecutive 5 evaluations, decrease learning rate
					if ndcg_not_decrease_time == 5:
						lr_decrease_time += 1
						if lr_decrease_time == config.early_stopping_round:
							break
						else:
							optimizer.param_groups[0]['lr'] *= 0.5
							ndcg_not_decrease_time = 0
				print("epoch {} : hits@{} : {}, ndcg@{} : {}.".format(epoch, config.topn, hits, config.topn, best_ndcg))
				#if USE_CUDA:
					#model.cuda()

		elif (epoch + 1) % 10 == 0 or epoch == 0:
			torch.save(model, os.path.join('./model/' + args.dataset, filename))

	writeList = [filename, 
		'testSet', '%.6f' % hits, '%.6f' % best_ndcg,]

	# Write the result into file
	with open(os.path.join('./result/', args.dataset + '.txt'), 'a') as fw:
		fw.write('\t'.join(writeList) + '\n')