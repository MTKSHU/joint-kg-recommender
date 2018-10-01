#-*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import math
import pandas as pd
import multiprocessing
import time

class MyTopNEvalProcess(multiprocessing.Process):
    def __init__(self, L, queue=None):
        super(MyTopNEvalProcess, self).__init__()
        self.queue = queue
        self.L = L

    def run(self):
        while True:
            all_user_lists = self.queue.get()
            try:
                self.process_data(all_user_lists)
            except:
                time.sleep(5)
                self.process_data(all_user_lists)
            self.queue.task_done()

    def process_data(self, all_user_lists):
        for user_lists in all_user_lists:
            f, p, r, hit_ratio, ndcg = get_performance(user_lists[0], user_lists[1])
            self.L.append([f, p, r, hit_ratio, ndcg])

def get_performance(recommend_list, purchased_list):
    """计算F1值。
    输入：
        recommend_list：n * 1, 推荐算法给出的推荐结果。
        purchased_list：m * 1,  用户的真实购买记录。
    输出：
        F1值。
    """
    k = 0
    hit_number = 0
    rank_list = []
    for i_id in recommend_list:
        k += 1
        if i_id in purchased_list:
            hit_number += 1
            rank_list.append(1)
        else:
            rank_list.append(0)
    
    k_gold = len(purchased_list)
    f = 0.0
    p = 0.0
    r = 0.0
    ndcg = 0.0

    if hit_number > 0:
        p = float(hit_number) / k
        r = float(hit_number) / k_gold
        f = 2 * p * r / (p + r)
        ndcg = ndcg_at_k(rank_list, k)

    return f, p, r, 1 if hit_number > 0 else 0, ndcg

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def evalAll(recommend_list, purchased_list, num_processes=multiprocessing.cpu_count()):
    """计算F1和NDCG值。
    输入：
        recommend_list： 推荐算法给出的推荐结果。
        purchased_list： 用户的真实购买记录。
    输出：
        F1，NDCG值。
    """
    assert len(recommend_list) == len(purchased_list), "Eval user number not match!"
    user_num = len(recommend_list)
    eval_list = []
    user_per_process = math.ceil(float(user_num) / num_processes)
    
    for i in range(num_processes):
        eval_list.append(zip(recommend_list[i*user_per_process:(i+1)*user_per_process], purchased_list[i*user_per_process:(i+1)*user_per_process]))

    with multiprocessing.Manager() as manager:
        L = manager.list()

        queue = multiprocessing.JoinableQueue()
        workerList = []
        for _ in range(num_processes):
            worker = MyTopNEvalProcess(L, queue=queue)
            workerList.append(worker)
            worker.daemon = True
            worker.start()

        for eval_batch in eval_list:
            queue.put(eval_batch)

        queue.join()

        results = np.array(list(L))

        for worker in workerList:
            worker.terminate()
    # f1, prec, rec, hit_ratio, ndcg
    performance = results.mean(axis=0)
    return performance[0], performance[1], performance[2], performance[3], performance[4]

if __name__ == "__main__":
    a = np.random.randint(0, 10, size=(2, 3))
    b = np.random.randint(0, 10, size=(2, 4))
    print(a)
    print(b)
    f1, prec, rec, hit_ratio, ndcg = evalAll(a, b, num_processes=1)
    print("{},{},{},{},{}".format(f1, prec, rec, hit_ratio, ndcg))