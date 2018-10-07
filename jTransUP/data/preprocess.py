import numpy as np
import pandas as pd
import csv
import json
import os
import random
import math
from jTransUP.utils.misc import Triple, Rating

LOW_FREQ = 10

def splitRelationType(allTripleList):
    allHeadDict = {}
    allTailDict = {}
    for triple in allTripleList:
        tmp_head_set = allHeadDict.get( (triple.t, triple.r), set())
        tmp_head_set.add(triple.h)
        allHeadDict[(triple.t, triple.r)] = triple.h

        tmp_tail_set = allTailDict.get( (triple.h, triple.r), set() )
        tmp_tail_set.add(triple.t)
        allTailDict[(triple.h, triple.r)] = tmp_tail_set
        
    one2oneRelations = set()
    one2manyRelations = set()
    many2oneRelations = set()
    many2manyRelations = set()
    
    rel_head_count_dict = {}
    rel_tail_count_dict = {}
    for er in allHeadDict:
        tmp_rel_count_list = rel_head_count_dict.get(er[1], [])
        tmp_rel_count_list.append(len(allHeadDict[er]))
        rel_head_count_dict[er[1]] = tmp_rel_count_list
    
    for er in allTailDict:
        tmp_rel_count_list = rel_tail_count_dict.get(er[1], [])
        tmp_rel_count_list.append(len(allTailDict[er]))
        rel_tail_count_dict[er[1]] = tmp_rel_count_list
    
    for r in rel_head_count_dict:
        avg_head_num = round( float( sum(rel_head_count_dict[r]) ) / len(rel_head_count_dict[r]) )
        avg_tail_num = round( float( sum(rel_tail_count_dict[r]) ) / len(rel_tail_count_dict[r]) )
        if avg_head_num > 1 and avg_tail_num > 1 :
            many2manyRelations.add(r)
        elif avg_head_num > 1 and avg_tail_num == 1:
            many2oneRelations.add(r)
        elif avg_head_num == 1 and avg_tail_num > 1:
            one2manyRelations.add(r)
        elif avg_head_num == 1 and avg_tail_num == 1:
            one2oneRelations.add(r)
        else:
            raise NotImplementedError
    return one2oneRelations, one2manyRelations, many2oneRelations, many2manyRelations

def splitKGData(triple_list, train_ratio = 0.7, test_ratio = 0.1, is_shuffle=False, is_filter=True):
    # valid ratio could be 1-train_ratio-test_ratio, and maybe zero
    assert train_ratio > 0 and train_ratio < 1, "train ratio out of range!"
    assert test_ratio > 0 and test_ratio < 1, "test ratio out of range!"

    valid_ratio = 1 - train_ratio - test_ratio
    assert valid_ratio >= 0 and valid_ratio < 1, "valid ratio out of range!"

    train_ent_set = set()
    train_rel_set = set()

    if is_shuffle : random.shuffle(triple_list)

    n_total = len(triple_list)
    n_train = math.ceil(n_total * train_ratio)
    n_valid = math.ceil(n_total * valid_ratio) if valid_ratio > 0 else 0

    # in case of zero test item
    if n_train >= n_total:
        n_train = n_total - 1
        n_valid = 0
    elif n_train + n_valid >= n_total :
        n_valid = n_total - 1 - n_train

    tmp_train_list = [i for i in triple_list[0:n_train]]
    tmp_valid_list = [i for i in triple_list[n_train:n_train+n_valid]]
    tmp_test_list = [i for i in triple_list[n_train+n_valid:]]
    
    for triple in tmp_train_list:
        train_ent_set.add(triple[0])
        train_ent_set.add(triple[1])
        train_rel_set.add(triple[2])
    
    e_map = {}
    for index, ent in enumerate(train_ent_set):
        e_map[ent] = index
    r_map = {}
    for index, rel in enumerate(train_rel_set):
        r_map[rel] = index
    
    train_list = [Triple(e_map[triple[0]], e_map[triple[1]], r_map[triple[2]]) for triple in tmp_train_list]

    if is_filter:
        valid_list = [Triple(e_map[triple[0]], e_map[triple[1]], r_map[triple[2]]) for triple in tmp_valid_list if triple[0] in train_ent_set and triple[1] in train_ent_set and triple[2] in train_rel_set]

        test_list = [Triple(e_map[triple[0]], e_map[triple[1]], r_map[triple[2]]) for triple in tmp_test_list if triple[0] in train_ent_set and triple[1] in train_ent_set and triple[2] in train_rel_set]
    else:
        valid_list = [Triple(e_map[triple[0]], e_map[triple[1]], r_map[triple[2]]) for triple in tmp_valid_list ]

        test_list = [Triple(e_map[triple[0]], e_map[triple[1]], r_map[triple[2]]) for triple in tmp_test_list ]
    # valid list length may be zero
    return train_list, valid_list, test_list, e_map, r_map

def parseRT(json_dict, ent_set=None, rel_set=None):
    r = json_dict['p']['value']
    t_type = json_dict['o']['type']
    t = json_dict['o']['value']
    if t_type != 'uri' or \
     ( ent_set is not None and t not in ent_set ) or \
     ( rel_set is not None and r not in rel_set ) :
        return None
    return r, t

def parseHR(json_dict, ent_set=None, rel_set=None):
    r = json_dict['p']['value']
    h = json_dict['s']['value']
    if  (ent_set is not None and h not in ent_set) or \
     ( rel_set is not None and r not in rel_set ) :
        return None
    return h, r

def processKG(kg_path, ent_vocab=None, rel_vocab=None, hop=1, train_ratio=0.7, test_ratio=0.2, is_shuffle=False, is_filter=True):
    triple_list = []
    ent_dic = {}
    for i in range(hop):
        triple_file = os.path.join(kg_path, "kg_hop{}.dat".format(i))
        with open(triple_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split('\t')
                if len(line_split) < 3 or \
                ( ent_vocab is not None and line_split[0] not in ent_vocab ) :
                    continue
                e = line_split[0]

                count = ent_dic.get(e, 0)
                ent_dic[e] = count + 1

                head_json_list = json.loads(line_split[1])
                tail_json_list = json.loads(line_split[2])
                for head_json in head_json_list:
                    rt = parseRT(head_json, ent_set=ent_vocab, rel_set=rel_vocab)
                    if rt is None: continue
                    r, t = rt
                    count = ent_dic.get(t, 0)
                    ent_dic[t] = count + 1
                    triple_list.append( (e, t, r) )

                for tail_json in tail_json_list:
                    hr = parseHR(tail_json, ent_set=ent_vocab, rel_set=rel_vocab)
                    if hr is None: continue
                    h, r = hr
                    count = ent_dic.get(h, 0)
                    ent_dic[h] = count + 1
                    triple_list.append( (h, e, r) )

    filtered_triple_list = [triple for triple in triple_list if ent_dic.get(triple[0], 0) >=LOW_FREQ and ent_dic.get(triple[1], 0)>=LOW_FREQ ]

    train_list, valid_list, test_list, e_map, r_map = splitKGData(filtered_triple_list, train_ratio=train_ratio, test_ratio=test_ratio, is_shuffle=is_shuffle, is_filter=is_filter)

    return train_list, valid_list, test_list, e_map, r_map

def splitRatingData(user_dict, train_ratio = 0.7, test_ratio = 0.2, is_shuffle=False, is_filter=True):
    # valid ratio could be 1-train_ratio-test_ratio, and maybe zero
    
    assert train_ratio > 0 and train_ratio < 1, "train ratio out of range!"
    assert test_ratio > 0 and test_ratio < 1, "test ratio out of range!"

    valid_ratio = 1 - train_ratio - test_ratio
    assert valid_ratio >= 0 and valid_ratio < 1, "valid ratio out of range!"

    train_item_set = set()
    tmp_train_list = []
    tmp_valid_list = []
    tmp_test_list = []
    for user in user_dict:
        tmp_item_list = user_dict[user]

        n_items = len(tmp_item_list)
        n_train = math.ceil(n_items * train_ratio)
        n_valid = math.ceil(n_items * valid_ratio) if valid_ratio > 0 else 0
        # in case of zero test item
        if n_train >= n_items:
            n_train = n_items - 1
            n_valid = 0
        elif n_train + n_valid >= n_items :
            n_valid = n_items - 1 - n_train

        if is_shuffle : random.shuffle(tmp_item_list)
        
        for ir in tmp_item_list[0:n_train]:
            tmp_train_list.append( (user, ir[0], ir[1]) )
            train_item_set.add(ir[0])
        tmp_valid_list.extend([(user, ir[0], ir[1]) for ir in tmp_item_list[n_train:n_train+n_valid]])

        tmp_test_list.extend( [(user, ir[0], ir[1]) for ir in tmp_item_list[n_train+n_valid:]] )

    u_map = {}
    for index, user in enumerate(user_dict.keys()):
        u_map[user] = index
    i_map = {}
    for index, item in enumerate(train_item_set):
        i_map[item] = index

    train_list = [Rating(u_map[rating[0]], i_map[rating[1]], rating[2]) for rating in tmp_train_list]
    
    if is_filter:
        valid_list = [Rating(u_map[rating[0]], i_map[rating[1]], rating[2]) for rating in tmp_valid_list if rating[1] in train_item_set]

        test_list = [Rating(u_map[rating[0]], i_map[rating[1]], rating[2]) for rating in tmp_test_list if rating[1] in train_item_set]
    else:
        valid_list = [Rating(u_map[rating[0]], i_map[rating[1]], rating[2]) for rating in tmp_valid_list ]

        test_list = [Rating(u_map[rating[0]], i_map[rating[1]], rating[2]) for rating in tmp_test_list ]

    return train_list, valid_list, test_list, u_map, i_map

def processRating(rating_file, item_vocab=None, train_ratio = 0.7, test_ratio = 0.2, is_shuffle=False, is_filter=True):

    df = pd.read_csv(rating_file)
    df = df[['userId', 'itemId', 'rating']]
    df = df.values

    user_dict = dict()
    item_dict = dict()

    f_user_dict = dict()
    f_item_dict = dict()

    for line in df:
        u_id = int(line[0])
        i_id = int(line[1])
        r_score = int(line[2])

        if item_vocab is not None and i_id not in item_vocab : continue

        if u_id in user_dict:
            user_dict[u_id].append( (i_id, r_score) )
        else:
            user_dict[u_id] = [(i_id, r_score)]

        if i_id in item_dict.keys():
            item_dict[i_id].append( (u_id, r_score) )
        else:
            item_dict[i_id] = [(u_id, r_score)]

    while True:
        print(len(user_dict.keys()), len(item_dict.keys()))
        flag1, flag2 = True, True

        for u_id in user_dict.keys():
            pos_items = user_dict[u_id]
            valid_items = [idx for idx in pos_items if idx[0] in item_dict.keys()]

            if len(valid_items) >= LOW_FREQ:
                f_user_dict[u_id] = valid_items
            else:
                flag1 = False

        for i_id in item_dict.keys():
            pos_users = item_dict[i_id]
            valid_users = [udx for udx in pos_users if udx[0] in user_dict.keys()]

            if len(valid_users) >= LOW_FREQ:
                f_item_dict[i_id] = valid_users
            else:
                flag2 = False

        user_dict = f_user_dict.copy()
        item_dict = f_item_dict.copy()
        f_user_dict = {}
        f_item_dict = {}

        if flag1 and flag2:
            print('select done')
            break
    
    train_list, valid_list, test_list, u_map, i_map = splitRatingData(user_dict, train_ratio = train_ratio, test_ratio = test_ratio, is_shuffle=is_shuffle, is_filter=is_filter)

    return train_list, valid_list, test_list, u_map, i_map

'''
def loadRelationTypes(filename):
    one2oneRelations = set()
    one2manyRelations = set()
    many2oneRelations = set()
    many2manyRelations = set()
    with open(filename, 'r') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) < 2 : continue
            if line_split[0] == 'one2one':
                tmp_set = one2oneRelations
            elif line_split[0] == 'one2many':
                tmp_set = one2manyRelations
            elif line_split[0] == 'many2one':
                tmp_set = many2oneRelations
            elif line_split[0] == 'many2many':
                tmp_set = many2manyRelations
            else:
                raise NotImplementedError
            for r_str in line_split[1:]:
                r = int(r_str)
                tmp_set.add(r)

    return one2oneRelations, one2manyRelations, many2oneRelations, many2manyRelations

o2o, o2m, m2o, m2m = loadRelationTypes(relation_type_file)

def splitRelationType(full_file, test_file):
        allTripleList = train_list + valid_list + test_list
        one2oneRelations, one2manyRelations, many2oneRelations, many2manyRelations = splitRelationType(allTripleList)
        with open(relation_type_file, 'w') as fout:
            fout.write('one2one\t{}\n'.format('\t'.join([str(r) for r in one2oneRelations])))
            fout.write('one2many\t{}\n'.format('\t'.join([str(r) for r in one2manyRelations])))
            fout.write('many2one\t{}\n'.format('\t'.join([str(r) for r in many2oneRelations])))
            fout.write('many2many\t{}\n'.format('\t'.join([str(r) for r in many2manyRelations])))
'''