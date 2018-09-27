import numpy as np
import os

class Triple(object):
	def __init__(self, head, tail, relation):
		self.h = head
		self.t = tail
		self.r = relation

class Ratings(object):
	def __init__(self, user, item):
		self.u = user
		self.i = item

# Gets the number of users/items
def getAnythingTotal(inPath, fileName):
	line_count = 0
	with open(os.path.join(inPath, fileName), 'r') as fr:
		for line in fr:
			if len(line) > 0 : line_count += 1
	return line_count

def loadRatings(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fin:
        ratingTotal = 0
        ratingList = []
        for line in fin:
            line_split = line.split(' ')
            user = int(line_split[0])
            for sp in line_split[1:]:
                item = int(sp)
                ratingList.append(Ratings(user, item))
                ratingTotal += 1
    ratingDict = {}
    for rating in ratingList:
        tmp_item_set = ratingDict.get(rating.u, set())
        tmp_item_set.add(rating.i)
        ratingDict[rating.u] = tmp_item_set

    return ratingTotal, ratingList, ratingDict

# split all ratings for training, validation and testing
def splitRatings(ratingList, isShuffle=False):
	if isShuffle:
		random.shuffle(ratingList)
	
	test_num = len(ratingList) // 5
	valid_num = ( len(ratingList) - test_num ) // 5
	testList = ratingList[:test_num]
	trainList = ratingList[test_num:-valid_num]
	validList = ratingList[-valid_num:]
	return trainList, validList, testList

# for evaluation
def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
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


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

