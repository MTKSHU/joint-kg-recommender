import os
from jTransUP.data.preprocess import process
import math
import random
from copy import deepcopy

def preprocessData(rating_file, train_ratio = 0.7, test_ratio = 0.1, is_shuffle=False, is_filter=True):
    # valid ratio could be 1-train_ratio-test_ratio, and maybe zero
    user_dict, u_map_dict, i_map_dict = process(rating_file)
    
    assert train_ratio > 0 and train_ratio < 1, "train ratio out of range!"
    assert test_ratio > 0 and test_ratio < 1, "test ratio out of range!"

    valid_ratio = 1 - train_ratio - test_ratio
    assert valid_ratio >= 0 and valid_ratio < 1, "valid ratio out of range!"

    train_item_set = set()
    for u_id in user_dict:
        tmp_item_list = user_dict[u_id]

        n_items = len(tmp_item_list)
        n_train = math.ceil(n_items * train_ratio)
        n_valid = math.ceil(n_items * valid_ratio) if valid_ratio > 0 else 0

        if is_shuffle : random.shuffle(tmp_item_list)
        tmp_train_list = [i for i in tmp_item_list[0:n_train]]
        train_item_set.update(tmp_train_list)
        tmp_valid_list = [i for i in tmp_item_list[n_train:n_train+n_valid]]
        tmp_test_list = [i for i in tmp_item_list[n_train+n_valid:]]

        user_dict[u_id] = (tmp_train_list, tmp_valid_list, tmp_test_list)
    
    if is_filter:
        for u_id in user_dict:
            item_list = user_dict[u_id]
            valid_list = [i_id for i_id in item_list[1] if i_id in train_item_set]
            test_list = [i_id for i_id in item_list[2] if i_id in train_item_set]
            user_dict[u_id] = (item_list[0], valid_list, test_list)

    return user_dict, u_map_dict, i_map_dict


def loadRatings(fileName):
    with open(fileName, 'r') as fin:
        rating_total = 0
        ratingDict = {}
        for line in fin:
            line_split = line.split(' ')
            if len(line_split) < 2 : continue
            user = int(line_split[0])
            item_set = set([int(i) for i in line_split[1:]])
            ratingDict[user] = item_set
            rating_total += len(item_set)

    return rating_total, ratingDict

def load_data(data_path, logger=None):
    train_ratio = 0.7
    test_ratio = 0.2
    isShuffle = True
    is_filter = True

    train_file = os.path.join(data_path, "train.dat")
    test_file = os.path.join(data_path, "test.dat")
    valid_file = os.path.join(data_path, "valid.dat") if 1 - train_ratio - test_ratio != 0 else None

    u_map_file = os.path.join(data_path, "user_vocab.dat")
    i_map_file = os.path.join(data_path, "item_vocab.dat")

    if not os.path.exists(train_file) or not os.path.exists(test_file) or \
    not os.path.exists(u_map_file) or not os.path.exists(i_map_file):
        rating_file = os.path.join(data_path, "ratings.csv")
        assert os.path.exists(rating_file), "make sure {}, {}, {} and {} exists or {}!".format(train_file, test_file, u_map_file, i_map_file, rating_file)

        str_is_shuffle = "shuffle and split" if isShuffle else "split without shuffle"
        if logger is not None:
            logger.debug("{} {} for {:.1f} training, {:.1f} validation and {:.1f} testing!".format(
            str_is_shuffle, rating_file, train_ratio, 1-train_ratio-test_ratio, test_ratio
        ))

        user_dict, u_map_dict, i_map_dict = preprocessData(rating_file, train_ratio=train_ratio, test_ratio=test_ratio, is_shuffle=isShuffle, is_filter=is_filter)

        f_train = open(train_file, 'w')
        f_test = open(test_file, 'w')
        f_valid = open(valid_file, 'w') if valid_file is not None else None

        for u_id in user_dict:
            f_train.write(str(u_id) + ' ' + ' '.join([str(i) for i in user_dict[u_id][0]]) + '\n')
            f_test.write(str(u_id) + ' ' + ' '.join([str(i) for i in user_dict[u_id][2]]) + '\n')
            if f_valid is not None :
                f_valid.write(str(u_id) + ' ' + ' '.join([str(i) for i in user_dict[u_id][1]]) + '\n')
        f_train.close()
        f_test.close()
        if f_valid is not None : f_valid.close()

        fu = open(u_map_file, 'w')
        fi = open(i_map_file, 'w')

        for u_id in u_map_dict.keys():
            fu.write(str(u_id) + ' ' + str(u_map_dict[u_id]) + '\n')
        fu.close()

        for i_id in i_map_dict.keys():
            fi.write(str(i_id) + ' ' + str(i_map_dict[i_id]) + '\n')
        fi.close()

    trainTotal, trainDict = loadRatings(train_file)
    testTotal, testDict = loadRatings(test_file)
    
    validTotal = 0
    validDict = None
    if valid_file is not None and os.path.exists(valid_file):
        validTotal, validDict = loadRatings(valid_file)
    
    if logger is not None:
        logger.info("Totally {} train, {} test and {} valid ratings!".format(trainTotal, testTotal, validTotal))
    
    # get item total
    item_total = 0
    with open(i_map_file, 'r') as fin:
        for line in fin:
            if len(line) > 0:
                item_total += 1
    # get user total
    user_total = 0
    with open(u_map_file, 'r') as fin:
        for line in fin:
            if len(line) > 0:
                user_total += 1
    
    if logger is not None:
        logger.info("successfully load {} users and {} items!".format(user_total, item_total))
    
    allRatingDict = deepcopy(trainDict)
    
    for u in testDict:
        item_set = allRatingDict.get(u, set())
        item_set |= testDict[u]
        allRatingDict[u] = item_set

    if validDict is not None:
        for u in validDict:
            item_set = allRatingDict.get(u, set())
            item_set |= validDict[u]
            allRatingDict[u] = item_set

    return trainDict, testDict, validDict, allRatingDict, user_total, item_total, trainTotal, testTotal, validTotal

if __name__ == "__main__":
    # Demo:
    trainDict, testDict, validDict, allRatingDict, user_total, item_total, trainTotal, testTotal, validTotal = load_data("/Users/caoyixin/Github/joint-kg-recommender/datasets/ml1m/")
    print("user:{}, item:{}!".format(user_total, item_total))
    print("totally ratings for {} train, {} valid, and {} test!".format(trainTotal, validTotal, testTotal))
    import random
    u_id = random.randrange(user_total)
    print("u:{} has brought items for train {}, valid {} and test {}!".format(u_id, trainDict[u_id], validDict[u_id] if validDict is not None else [], testDict[u_id]))