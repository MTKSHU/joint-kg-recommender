import math
from copy import deepcopy
import random

# Split the tripleList into #num_batches batches
def getBatchList(sampleList, batch_size):
	num_batches = math.ceil( len(sampleList) / batch_size )
	batchList = [0] * num_batches
	for i in range(num_batches - 1):
		batchList[i] = sampleList[i * batch_size : (i + 1) * batch_size]
	batchList[num_batches - 1] = sampleList[(num_batches - 1) * batch_size : ]
	return batchList

# generate the negtive samples for test ratings
def getTestNegList(testList, allRatingDict, itemTotal, neg_test_samples=100):
	negList = []
	has_user_set = set()
	for test_sample in testList:
		if test_sample.u in has_user_set : continue
		for i in range(neg_test_samples):
			tmp_rating = corrupt_item(test_sample, itemTotal, ratingDict=allRatingDict)
			negList.append(tmp_rating)
			tmp_item_set = allRatingDict.get(tmp_rating.u, set())
			tmp_item_set.add(tmp_rating.i)
			allRatingDict[tmp_rating.u] = tmp_item_set
		has_user_set.add(test_sample.u)
	return negList

# Change the user randomly
def corrupt_user(rating, userTotal, ratingDict=None):
	newRating = deepcopy(rating)
	oldUser = rating.u
	while True:
		newUser = random.randrange(userTotal)
		if ( ratingDict is None and newUser != oldUser ) or (
			ratingDict is not None and newUser in ratingDict and newRating.i in ratingDict[newUser] ):
			break
	newRating.u = newUser
	return newRating

# Change the item randomly
def corrupt_item(rating, itemTotal, ratingDict=None):
	newRating = deepcopy(rating)
	oldItem = rating.i
	item_set = ratingDict[rating.u] if ratingDict is not None and rating.u in ratingDict else set()
	while True:
		newItem = random.randrange(itemTotal)
		if ( ratingDict is None and newItem != oldItem ) or newItem in item_set :
			break
	newRating.i = newItem
	return newRating

def getRatingElements(ratingList):
	userList = [rating.u for rating in ratingList]
	itemList = [rating.i for rating in ratingList]
	return userList, itemList

# Use all the ratingList,
# and generate negative samples by corrupting user or item with equal probabilities,
# with checking whether false negative samples exist or not.
def getRatingAll(ratingList, userTotal, itemTotal, ratingDict=None):
	newRatingList = [corrupt_user(rating, userTotal, ratingDict=ratingDict) if random.random() < 0.5 
		else corrupt_item(rating, itemTotal, ratingDict=ratingDict) for rating in ratingList]
	pu, pi = getRatingElements(ratingList)
	nu, ni = getRatingElements(newRatingList)
	return pu, pi, nu, ni

# Sample a batch from ratingList,
# and generate negative samples by corrupting user or item with equal probabilities,
# with checking whether false negative samples exist or not.
def getRatingBatch(ratingList, batchSize, userTotal, itemTotal, ratingDict=None):
	oldRatingList = random.sample(ratingList, batchSize)
	newRatingList = [corrupt_user(rating, userTotal, ratingDict=ratingDict) if random.random() < 0.5 
		else corrupt_item(rating, itemTotal, ratingDict=ratingDict) for rating in oldRatingList]
	pu, pi = getRatingElements(oldRatingList)
	nu, ni = getRatingElements(newRatingList)
	return pu, pi, nu, ni