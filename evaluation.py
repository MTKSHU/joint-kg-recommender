#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-15 16:03:42
# @Author  : jimmy (jimmywangheng@qq.com)
# @Link    : http://sdcs.sysu.edu.cn
# @Version : $Id$

import os
import numpy as np
import time
import datetime
import random
import multiprocessing
import math

from evaluation_onehot import eval_model_pro
from data import getBatchList

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from projection import *

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor

def isHit10(triple, tree, cal_embedding, tripleDict, isTail):
	# If isTail == True, evaluate the prediction of tail entity
	if isTail == True:
		k = 0
		wrongCount = 0
		while wrongCount < 10:
			k += 15
			tail_dist, tail_ind = tree.query(cal_embedding, k=k)
			for elem in tail_ind[0][k - 15: k]:
				if triple.t == elem:
					return True
				elif (triple.h, elem, triple.r) in tripleDict:
					continue
				else:
					wrongCount += 1
					if wrongCount > 9:
						return False
	# If isTail == False, evaluate the prediction of head entity
	else:
		k = 0
		wrongCount = 0
		while wrongCount < 10:
			k += 15
			head_dist, head_ind = tree.query(cal_embedding, k=k)
			for elem in head_ind[0][k - 15: k]:
				if triple.h == elem:
					return True
				elif (elem, triple.t, triple.r) in tripleDict:
					continue
				else:
					wrongCount += 1
					if wrongCount > 9:
						return False

# Find the rank of ground truth tail in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereTail(head, tail, rel, array, tripleDict):
	wrongAnswer = 0
	for num in array:
		if num == tail:
			return wrongAnswer
		elif (head, num, rel) in tripleDict:
			continue
		else:
			wrongAnswer += 1
	return wrongAnswer

# Find the rank of ground truth head in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereHead(head, tail, rel, array, tripleDict):
	wrongAnswer = 0
	for num in array:
		if num == head:
			return wrongAnswer
		elif (num, tail, rel) in tripleDict:
			continue
		else:
			wrongAnswer += 1
	return wrongAnswer

def pairwise_L1_distances(A, B):
	dist = torch.sum(torch.abs(A.unsqueeze(1) - B.unsqueeze(0)), dim=2)
	return dist

def pairwise_L2_distances(A, B):
	AA = torch.sum(A ** 2, dim=1).unsqueeze(1)
	BB = torch.sum(B ** 2, dim=1).unsqueeze(0)
	dist = torch.mm(A, torch.transpose(B, 0, 1))
	dist *= -2
	dist += AA
	dist += BB
	return dist

	

def evaluation_transUP_helper(testList, user_embeddings, item_embeddings, pref_embeddings, norm_embeddings, L1_flag):
	# embeddings are torch tensor like (No Variable!)
	# Only one kind of relation

	userList = [rating.u for rating in testList]
	itemList = [rating.i for rating in testList]

	u_e = user_embeddings[userList]
	i_e = item_embeddings[itemList]

	# use item and user embedding to compute preference distribution
	pre_probs = torch.matmul(u_e + i_e, torch.t(pref_embeddings)) / 2
	r_e = torch.matmul(pre_probs, pref_embeddings)
	norm_e = torch.matmul(pre_probs, norm_embeddings)

	proj_u_e = projection_transH_pytorch(u_e, norm_e)
	proj_i_e = projection_transH_pytorch(i_e, norm_e)

	if L1_flag:
		predicted_scores = torch.sum(torch.abs(proj_u_e + r_e - proj_i_e), 1)
	else:
		predicted_scores = torch.sum((proj_u_e + r_e - proj_i_e) ** 2, 1)
	
	predicted_scores = predicted_scores.cpu().numpy()

	return userList, itemList, predicted_scores

class MyProcessTransUP(multiprocessing.Process):
	def __init__(self, L, user_embeddings, item_embeddings, pref_embeddings, norm_embeddings, L1_flag, queue=None):
		super(MyProcessTransUP, self).__init__()
		self.L = L
		self.queue = queue
		self.user_embeddings = user_embeddings
		self.item_embeddings = item_embeddings
		self.pref_embeddings = pref_embeddings
		self.norm_embeddings = norm_embeddings
		self.L1_flag = L1_flag

	def run(self):
		while True:
			testList = self.queue.get()
			try:
				self.process_data(testList, self.user_embeddings, self.item_embeddings, self.pref_embeddings, self.norm_embeddings, self.L1_flag, self.L)
			except:
				time.sleep(5)
				self.process_data(testList, self.user_embeddings, self.item_embeddings, self.pref_embeddings, self.norm_embeddings, self.L1_flag, self.L)
			self.queue.task_done()

	def process_data(self, testList, user_embeddings, item_embeddings, pref_embeddings, norm_embeddings, L1_flag, L):

		userList, itemList, predictList = evaluation_transUP_helper(testList, user_embeddings, item_embeddings, pref_embeddings, norm_embeddings, L1_flag)

		L.append((userList, itemList, predictList))

def evaluation_transUP(testList, negTestList, user_embeddings, item_embeddings, pref_embeddings, norm_embeddings, L1_flag, batch_size, neg_samples, topn=10, num_processes=multiprocessing.cpu_count()):
	testBatchList = getBatchList(testList, batch_size)

	negResultDict = {}
	negBatchList = getBatchList(negTestList, batch_size)
	negResultList = evaluation_transUP_Batch(negBatchList, user_embeddings, item_embeddings, pref_embeddings, norm_embeddings, L1_flag, num_processes=num_processes)
	for elem in negResultList:
		for i, u in enumerate(elem[0]):
			tmpNegUserResult = negResultDict.get(u, [])
			tmpNegUserResult.append(elem[2][i])
			negResultDict[u] = tmpNegUserResult

	posResultList = evaluation_transUP_Batch(testBatchList, user_embeddings, item_embeddings, pref_embeddings, norm_embeddings, L1_flag, num_processes=num_processes)
	resultList = []
	for elem in posResultList:
		for i, u in enumerate(elem[0]):
			u_neg_result = negResultDict[u]
			resultList += [elem[2][i]] + u_neg_result

	y_gnd_one = [1] + [0] * neg_samples
	y_gnd = y_gnd_one * len(testList)
	
	hits, ndcg = eval_model_pro(y_gnd, resultList, topn, neg_samples+1)

	return hits, ndcg
	
# for each rating in testBatchList, compute a score, and return the score list
def evaluation_transUP_Batch(testBatchList, user_embeddings, item_embeddings, pref_embeddings, norm_embeddings, L1_flag, num_processes=multiprocessing.cpu_count()):
	# embeddings are torch tensor like (No Variable!)

	user_embeddings = user_embeddings.cpu()
	item_embeddings = item_embeddings.cpu()
	pref_embeddings = pref_embeddings.cpu()
	norm_embeddings = norm_embeddings.cpu()

	with multiprocessing.Manager() as manager:
		L = manager.list()
		queue = multiprocessing.JoinableQueue()
		workerList = []
		for i in range(num_processes):
			worker = MyProcessTransUP(L, user_embeddings, item_embeddings, pref_embeddings, norm_embeddings, L1_flag, queue=queue)
			workerList.append(worker)
			worker.daemon = True
			worker.start()
		for batch in testBatchList:
			queue.put(batch)

		queue.join()

		resultList = list(L)

		for worker in workerList:
			worker.terminate()
		
		return resultList