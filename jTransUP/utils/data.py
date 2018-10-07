import random
from copy import deepcopy
import numpy as np
from jTransUP.utils.misc import Triple, Rating

def getEvalRatingBatch(rating_batch):
    u_ids = [rating.u for rating in rating_batch]
    i_ids = [rating.i for rating in rating_batch]
    return u_ids, i_ids

def getTrainRatingBatch(rating_batch, item_total, allDict):
    u_ids = [rating.u for rating in rating_batch]
    pi_ids = [rating.i for rating in rating_batch]
    # yield u, pi, ni, each list contains batch size ids,
    u, pi, ni = addNegRatings(rating_batch.tolist(), item_total, ratingDict=allDict)
    return u, pi, ni

def getTrainTripleBatch(triple_batch, entity_total, allHeadDict, allTailDict):
    negTripleList = [corrupt_head_filter(triple, entity_total, allHeadDict) if random.random() < 0.5 
		else corrupt_tail_filter(triple, entity_total, allTailDict) for triple in triple_batch]
    # yield u, pi, ni, each list contains batch size ids,
    ph, pt, pr = getTripleElements(triple_batch)
    nh, nt, nr = getTripleElements(negTripleList)
    return ph, pt, pr, nh, nt, nr

# Change the head of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_head_filter(triple, entityTotal, tripleDict):
	newTriple = deepcopy(triple)
	while True:
		newHead = random.randrange(entityTotal)
		if (newTriple.t, newTriple.r) not in tripleDict or \
         ( (newTriple.t, newTriple.r) in tripleDict and newHead not in tripleDict[(newTriple.t, newTriple.r)] ):
			break
	newTriple.h = newHead
	return newTriple

# Change the tail of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_tail_filter(triple, entityTotal, tripleDict):
	newTriple = deepcopy(triple)
	while True:
		newTail = random.randrange(entityTotal)
		if (newTriple.h, newTriple.r) not in tripleDict or \
        ( (newTriple.h, newTriple.r) in tripleDict and newTail not in tripleDict[(newTriple.h, newTriple.r)] ):
			break
	newTriple.t = newTail
	return newTriple

def getTripleElements(tripleList):
	headList = [triple.h for triple in tripleList]
	tailList = [triple.t for triple in tripleList]
	relList = [triple.r for triple in tripleList]
	return headList, tailList, relList

# Use all the tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# with checking whether false negative samples exist.
def getNegTriples(tripleList, entityTotal, allHeadDict, allTailDict):
	newTripleList = [corrupt_head_filter(triple, entityTotal, allHeadDict) if random.random() < 0.5 
		else corrupt_tail_filter(triple, entityTotal, allTailDict) for triple in tripleList]
	return newTripleList

def addNegRatings(ratingList, itemTotal, ratingDict={}):
    ni = []
    neg_set = set()
    for rating in ratingList:
        oldItem = rating.i
        item_set = ratingDict[rating.u] if rating.u in ratingDict else set()
        while True:
            newItem = random.randrange(itemTotal)
            if newItem != oldItem and newItem not in item_set and newItem not in neg_set :
                break
        ni.append(newItem)
        neg_set.add(newItem)
    u = [rating.u for rating in ratingList]
    pi = [rating.i for rating in ratingList]
    return u, pi, ni

def MakeTrainIterator(
        trainList,
        batch_size,
        negtive_samples=1):
    train_list = np.array(trainList)

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
            
            # numpy
            yield train_list[batch_indices]
        
    return data_iter()

def MakeRatingEvalIterator(
        evalDict,
        itemTotal,
        batch_size,
        allDict={}):
    # Make a list of minibatches from a dataset to use as an iterator.

    def data_iter():
        batch_ratings = []
        while True:
            for u in evalDict:
                if len(evalDict[u]) < 1 : continue
                start = 0
                filter_set = allDict[u] if u in allDict else set()
                item_set = set(range(itemTotal)) - filter_set

                item_list = list(item_set|evalDict[u])
                while True:
                    end = batch_size - len(batch_ratings)
                    if start + end >= len(item_list) :
                        batch_ratings.extend( [Rating(u, i, 1) for i in item_list[start:] ])
                        break
                    batch_ratings.extend( [Rating(u, i, 1) for i in item_list[start : start + end] ])
                    start += end
                    yield batch_ratings
                    del batch_ratings[:]
            if len(batch_ratings) > 0 :
                yield batch_ratings
                del batch_ratings[:]
            yield None

    return data_iter()

def MakeTripleEvalIterator(
        evalList,
        entTotal,
        batch_size,
        allDict={},
        isHead=True):
    # Make a list of minibatches from a dataset to use as an iterator.
    evalDict = {}

    for triple in evalList:
        if isHead:
            tmp_head_set = evalDict.get( (triple.t, triple.r), set())
            tmp_head_set.add(triple.h)
            evalDict[(triple.t, triple.r)] = tmp_head_set
        else:
            tmp_tail_set = evalDict.get( (triple.h, triple.r), set() )
            tmp_tail_set.add(triple.t)
            evalDict[(triple.h, triple.r)] = tmp_tail_set

    # iteratively generate all eval triples
    # yield None : terminate
    def data_iter():
        batch_triples = []
        while True:
            for er in evalDict:
                start = 0
                filter_set = allDict[er] if er in allDict else set()
                ent_set = set(range(entTotal)) - filter_set

                ent_list = list(ent_set|evalDict[er])
                while True:
                    end = batch_size - len(batch_triples)
                    if start + end >= len(ent_list) :
                        for e in ent_list[start:]:
                            tmp_triple = Triple(e, er[0], er[1]) if isHead else Triple(er[0], e, er[1])
                            batch_triples.append(tmp_triple)
                        break
                    for e in ent_list[start : start + end]:
                        tmp_triple = Triple(e, er[0], er[1]) if isHead else Triple(er[0], e, er[1])
                        batch_triples.append(tmp_triple)
                    start += end
                    
                    yield batch_triples
                    del batch_triples[:]
            if len(batch_triples) > 0 :
                yield batch_triples
                del batch_triples[:]
            yield None

    return data_iter()