import torch
from collections import deque
import numpy as np
from jTransUP.utils.evaluation import evalAll
import heapq
import time
from itertools import groupby

USE_CUDA = torch.cuda.is_available()

def to_gpu(var):
    if USE_CUDA:
        return var.cuda()
    return var

def projection_transH_pytorch(original, norm):
    return original - torch.sum(original * norm, dim=1, keepdim=True) * norm

class Triple(object):
	def __init__(self, head, tail, relation):
		self.h = head
		self.t = tail
		self.r = relation

class Rating(object):
	def __init__(self, user, item, rating):
		self.u = user
		self.i = item
		self.r = rating

class Accumulator(object):
    """Accumulator. Makes it easy to keep a trailing list of statistics."""

    def __init__(self, maxlen=None):
        self.maxlen = maxlen
        self.cache = dict()

    def add(self, key, val):
        self.cache.setdefault(key, deque(maxlen=self.maxlen)).append(val)

    def get(self, key, clear=True):
        ret = self.cache.get(key, [])
        if clear:
            try:
                del self.cache[key]
            except BaseException:
                pass
        return ret

    def get_avg(self, key, clear=True):
        return np.array(self.get(key, clear)).mean()

# x[0] is the identity, x[1] x[-1] is score
def evalProcess(test_list, pred_dict, is_descending=True):
    grouped = [(identity, list(g)) for identity, g in groupby(test_list, key=lambda x:x[0])]
    for identity, subList in grouped:
        full_list = pred_dict.get(identity, []) + subList
        full_list.sort(key=lambda x:x[-1], reverse=is_descending)
        pred_dict[identity] = full_list
    return pred_dict

def getPerformance(predDict, testDict, topn=10):
    pred_list = []
    gold_list = []

    mean_rank_list = []
    for identity in predDict:
        if identity not in testDict : continue
        sub_pred_list = [rank_tuple[1] for rank_tuple in predDict[identity]]
        sub_gold_list = list(testDict[identity])

        # mean rank
        rank_indexes = [sub_pred_list.index(y)+1 for y in sub_gold_list]
        
        mean_rank_list.append( float(sum(rank_indexes) - (len(rank_indexes) * (len(rank_indexes) - 1 )/2.0) ) / len(rank_indexes) )
        
        if topn > 0:
            sub_pred_list = sub_pred_list[:topn]
            sub_gold_list = sub_gold_list[:topn]
        pred_list.append(sub_pred_list)
        gold_list.append(sub_gold_list)
    f1, p, r, hit, ndcg = evalAll(pred_list, gold_list)
    
    mean_rank = float( sum(mean_rank_list) ) / len(mean_rank_list)

    return f1, p, r, hit, ndcg, mean_rank

def recursively_set_device(inp, gpu=USE_CUDA):
    if hasattr(inp, 'keys'):
        for k in list(inp.keys()):
            inp[k] = recursively_set_device(inp[k], USE_CUDA)
    elif isinstance(inp, list):
        return [recursively_set_device(ii, USE_CUDA) for ii in inp]
    elif isinstance(inp, tuple):
        return (recursively_set_device(ii, USE_CUDA) for ii in inp)
    elif hasattr(inp, 'cpu'):
        if USE_CUDA:
            inp = inp.cuda()
        else:
            inp = inp.cpu()
    return inp