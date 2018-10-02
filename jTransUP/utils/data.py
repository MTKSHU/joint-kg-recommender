import random
from copy import deepcopy
import numpy as np

def addNegRatings(ratingList, itemTotal, ratingDict=None):
    ni = []
    neg_set = set()
    for rating in ratingList:
        oldItem = rating[1]
        item_set = ratingDict[rating[0]] if ratingDict is not None and rating[0] in ratingDict else set()
        while True:
            newItem = random.randrange(itemTotal)
            if newItem != oldItem and newItem not in item_set and newItem not in neg_set :
                break
        ni.append(newItem)
        neg_set.add(newItem)
    u = [rating[0] for rating in ratingList]
    pi = [rating[1] for rating in ratingList]
    return u, pi, ni

def MakeTrainIterator(
        trainDict,
        item_total,
        batch_size,
        negtive_samples=1,
        allRatingDict=None):
    train_list = []
    for u_id in trainDict:
        for i_id in trainDict[u_id]:
            train_list.append((u_id, i_id))
    train_list = np.array(train_list)

    def data_iter():
        dataset_size = len(train_list)
        order = list(range(dataset_size)) * negtive_samples
        random.shuffle(order)
        start = -1 * batch_size

        while True:
            start += batch_size
            if start > dataset_size - batch_size:
                # Start another epoch.
                start = 0
                random.shuffle(order)
            batch_indices = order[start:start + batch_size]
            # yield u, pi, ni, each list contains batch size ids,
            u, pi, ni = addNegRatings(train_list[batch_indices].tolist(), item_total, ratingDict=allRatingDict)
            yield u, pi, ni
        
    return data_iter()

def MakeEvalIterator(
        evalDict,
        itemTotal,
        batch_size,
        allRatingDict=None):
    # Make a list of minibatches from a dataset to use as an iterator.
    def data_iter():
        u_list = []
        i_list = []
        while True:
            for u in evalDict:
                start = 0
                filter_set = allRatingDict[u] if allRatingDict is not None and u in allRatingDict else set()
                item_set = set(range(itemTotal)) - filter_set

                item_list = list(item_set|evalDict[u])
                while True:
                    end = batch_size - len(i_list)
                    if start + end > len(item_list) :
                        i_list.extend(item_list[start :])
                        u_list.extend([u] * len(i_list))
                        break
                    u_list.extend([u]*end)
                    i_list.extend(item_list[start : start + end])
                    start += end
                    yield u_list, i_list
                    del u_list[:]
                    del i_list[:]
            if len(i_list) > 0 :
                yield u_list, i_list
                del u_list[:]
                del i_list[:]
            yield None

    return data_iter()