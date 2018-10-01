import torch
from collections import deque
import numpy as np
from jTransUP.utils.evaluation import evalAll
import heapq
import multiprocessing
import time
from itertools import groupby

USE_CUDA = torch.cuda.is_available()

def to_gpu(var):
    if USE_CUDA:
        return var.cuda()
    return var

def projection_transH_pytorch(original, norm):
    return original - torch.sum(original * norm, dim=1, keepdim=True) * norm

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

class MyEvalProcess(multiprocessing.Process):
    def __init__(self, pred_dict, lock, topn=10, target=1, queue=None):
        super(MyEvalProcess, self).__init__()
        self.queue = queue
        self.pred_dict = pred_dict
        self.topn = topn
        self.lock = lock
        self.rank_f = heapq.nlargest if target == 1 else heapq.nsmallest

    def run(self):
        while True:
            ratings = self.queue.get()
            try:
                self.process_data(list(ratings))
            except:
                time.sleep(5)
                self.process_data(list(ratings))
            self.queue.task_done()

    def process_data(self, test_list):
        grouped = [(u_id, list(g)) for u_id, g in groupby(test_list, key=lambda x:x[0])]
        ranked_list = []
        for u_id, subList in grouped:
            ranked_sublist = self.rank_f(self.topn, subList, key=lambda x:x[2])
            ranked_list.append((u_id, ranked_sublist))
        
        self.lock.acquire()
        for u_list in ranked_list:
            if u_list[0] not in self.pred_dict:
                self.pred_dict[u_list[0]] = u_list[1]
            elif u_list[0] in self.pred_dict :
                new_list = self.pred_dict[u_list[0]] + u_list[1]
                ranked_newlist = self.rank_f(self.topn, new_list, key=lambda x:x[2])
                self.pred_dict[u_list[0]] = ranked_newlist
        self.lock.release()

def getPerformance(predDict, testDict, num_processes=multiprocessing.cpu_count()):
    pred_list = []
    gold_list = []
    
    for u_id in predDict:
        if u_id not in testDict : continue
        pred_list.append([rating[1] for rating in predDict[u_id]])
        gold_list.append(list(testDict[u_id]))
    
    f1, hit, ndcg, p, r = evalAll(pred_list, gold_list, num_processes=num_processes)
    return f1, hit, ndcg, p, r

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