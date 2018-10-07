import os
from jTransUP.data.preprocess import processRating
from jTransUP.utils.data import MakeTrainIterator, MakeRatingEvalIterator
import math
import random
from copy import deepcopy
from jTransUP.utils.misc import Rating

def loadVocab(filename):
    with open(filename, 'r') as fin:
        vocab_total = 0
        vocab = {}
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 2 : continue
            mapped_id = int(line_split[0])
            org_id = int(line_split[1])
            vocab[org_id] = mapped_id
            vocab_total += 1

    return vocab_total, vocab

def loadRatingList(filename):
    with open(filename, 'r') as fin:
        rating_total = 0
        rating_list = []
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 3 : continue
            u_id = int(line_split[0])
            i_id = int(line_split[1])
            r_score = int(line_split[2])
            rating_list.append(Rating(u_id, i_id, r_score))
            rating_total += 1

    return rating_total, rating_list

def loadRatingDict(filename):
    with open(filename, 'r') as fin:
        rating_total = 0
        rating_dict = {}
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 3 : continue
            u_id = int(line_split[0])
            i_id = int(line_split[1])
            r_score = int(line_split[2])
            tmp_items = rating_dict.get(u_id, set())
            tmp_items.add(i_id)
            rating_dict[u_id] = tmp_items
            rating_total += 1

    return rating_total, rating_dict

def load_data(data_path, batch_size, filter_wrong_corrupted=True, item_vocab=None, logger=None, filter_unseen_samples=True, shuffle_data_split=False, train_ratio=0.7, test_ratio=0.2):

    train_file = os.path.join(data_path, "train.dat")
    test_file = os.path.join(data_path, "test.dat")
    valid_file = os.path.join(data_path, "valid.dat") if 1 - train_ratio - test_ratio != 0 else None

    u_map_file = os.path.join(data_path, "u_map.dat")
    i_map_file = os.path.join(data_path, "i_map.dat")

    if not os.path.exists(train_file) or not os.path.exists(test_file) or \
    not os.path.exists(u_map_file) or not os.path.exists(i_map_file):
        rating_file = os.path.join(data_path, "ratings.csv")
        assert os.path.exists(rating_file), "make sure {}, {}, {} and {} exists or {}!".format(train_file, test_file, u_map_file, i_map_file, rating_file)

        str_is_shuffle = "shuffle and split" if shuffle_data_split else "split without shuffle"
        if logger is not None:
            logger.debug("{} {} for {:.1f} training, {:.1f} validation and {:.1f} testing!".format(
            str_is_shuffle, rating_file, train_ratio, 1-train_ratio-test_ratio, test_ratio
        ))

        train_list, valid_list, test_list, u_map, i_map = processRating(rating_file, item_vocab=item_vocab, train_ratio = train_ratio, test_ratio = test_ratio, is_shuffle=shuffle_data_split, is_filter=filter_unseen_samples)

        # save ent_dic, rel_dic 
        with open(u_map_file, 'w') as fout:
            for org_u_id in u_map:
                fout.write('{}\t{}\n'.format(u_map[org_u_id], org_u_id))
        with open(i_map_file, 'w') as fout:
            for org_i_id in i_map:
                fout.write('{}\t{}\n'.format(i_map[org_i_id], org_i_id))
        with open(train_file, 'w') as fout:
            for rating in train_list:
                fout.write('{}\t{}\t{}\n'.format(rating.u, rating.i, rating.r))
        with open(test_file, 'w') as fout:
            for rating in test_list:
                fout.write('{}\t{}\t{}\n'.format(rating.u, rating.i, rating.r))
        
        if len(valid_list) > 0:
            with open(valid_file, 'w') as fout:
                for rating in valid_list:
                    fout.write('{}\t{}\t{}\n'.format(rating.u, rating.i, rating.r))

    trainTotal, trainList = loadRatingList(train_file)
    testTotal, testDict = loadRatingDict(test_file)
    validTotal = 0
    validDict = {}
    if valid_file is not None and os.path.exists(valid_file):
        validTotal, validDict = loadRatingDict(valid_file)
    
    if logger is not None:
        logger.info("Totally {} train, {} test and {} valid ratings!".format(trainTotal, testTotal, validTotal))
    
    # get user total
    user_total, u_map = loadVocab(u_map_file)
    # get relation total
    item_total, i_map = loadVocab(i_map_file)
    
    if logger is not None:
        logger.info("successfully load {} users and {} items!".format(user_total, item_total))
    
    train_iter = MakeTrainIterator(trainList, batch_size, negtive_samples=1)

    allDict = {}
    if filter_wrong_corrupted:
        allDict = deepcopy(testDict)
        if validTotal > 0 :
            for u_id in validDict:
                tmp_items = allDict.get(u_id, set())
                tmp_items |= validDict[u_id]
                allDict[u_id] = tmp_items
        for rating in trainList:
            tmp_items = allDict.get(rating.u, set())
            tmp_items.add(rating.i)
            allDict[rating.u] = tmp_items

    test_iter = MakeRatingEvalIterator(testDict, item_total, batch_size, allDict=allDict)

    valid_iter = None
    if validTotal > 0:
        valid_iter = MakeRatingEvalIterator(validDict, item_total, batch_size, allDict=allDict)

    datasets = (trainList, testDict, validDict, allDict, testTotal, validTotal)
    rating_iters = (train_iter, test_iter, valid_iter)

    return datasets, rating_iters, u_map, i_map, user_total, item_total

if __name__ == "__main__":
    # Demo:
    data_path = "/Users/caoyixin/Github/joint-kg-recommender/datasets/ml1m/"
    batch_size = 10
    from jTransUP.data.load_kg_rating_data import loadR2KgMap

    i2kg_file = os.path.join(data_path, 'i2kg_map.tsv')
    i2kg_pairs = loadR2KgMap(i2kg_file)
    i_set = set([p[0] for p in i2kg_pairs])

    datasets, rating_iters, u_map, i_map, user_total, item_total = load_data(data_path, batch_size, item_vocab=i_set)

    trainList, testDict, validDict, allDict, testTotal, validTotal = datasets
    print("user:{}, item:{}!".format(user_total, item_total))
    print("totally ratings for {} train, {} valid, and {} test!".format(len(trainList), item_total, testTotal))