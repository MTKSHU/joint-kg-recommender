import os
from jTransUP.data.preprocess import processKG, splitRelationType
import math
import random
from copy import deepcopy
from jTransUP.utils.misc import Triple
from jTransUP.utils.data import MakeTrainIterator, MakeTripleEvalIterator

def loadTriples(filename):
    with open(filename, 'r') as fin:
        triple_total = 0
        triple_list = []
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 3 : continue
            h_id = int(line_split[0])
            t_id = int(line_split[1])
            r_id = int(line_split[2])
            triple_list.append(Triple(h_id, t_id, r_id))
            triple_total += 1

    return triple_total, triple_list

def loadVocab(filename):
    with open(filename, 'r') as fin:
        vocab_total = 0
        vocab = {}
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 2 : continue
            e_id = int(line_split[0])
            e_uri = line_split[1]
            vocab[e_uri] = e_id
            vocab_total += 1

    return vocab_total, vocab

def buildAllDict(allTripleList):
    allHeadDict = {}
    allTailDict = {}
    for triple in allTripleList:
        tmp_head_set = allHeadDict.get( (triple.t, triple.r), set())
        tmp_head_set.add(triple.h)
        allHeadDict[(triple.t, triple.r)] = tmp_head_set
        tmp_tail_set = allTailDict.get( (triple.h, triple.r), set() )
        tmp_tail_set.add(triple.t)
        allTailDict[(triple.h, triple.r)] = tmp_tail_set
    return allHeadDict, allTailDict

def getEvalIter(eval_list, entity_total, batch_size, allHeadDict=None, allTailDict=None):

    head_iter = MakeTripleEvalIterator(eval_list, entity_total, batch_size, allDict=allHeadDict, isHead=True)

    tail_iter = MakeTripleEvalIterator(eval_list, entity_total, batch_size, allDict=allTailDict, isHead=False)

    return head_iter, tail_iter

def load_data(kg_path, batch_size, filter_wrong_corrupted=True, rel_vocab=None, logger=None, hop=1, filter_unseen_samples=True, shuffle_data_split=False, train_ratio=0.7, test_ratio=0.2):

    # each dataset has the /kg/ dictionary

    train_file = os.path.join(kg_path, "train.dat")
    test_file = os.path.join(kg_path, "test.dat")
    valid_file = os.path.join(kg_path, "valid.dat") if 1 - train_ratio - test_ratio != 0 else None

    e_map_file = os.path.join(kg_path, "e_map.dat")
    r_map_file = os.path.join(kg_path, "r_map.dat")

    relation_type_file = os.path.join(kg_path, "relation_type.dat")

    # could be no validation file
    if not os.path.exists(train_file) or not os.path.exists(test_file) or \
    not os.path.exists(e_map_file) or not os.path.exists(r_map_file) :
        for i in range(hop):
            triple_file = os.path.join(kg_path, "kg_hop{}.dat".format(i))

            assert os.path.exists(triple_file), "make sure {}, {}, {}, {} and {} exists or provide raw data: {}!".format(train_file, test_file, e_map_file, r_map_file, relation_type_file, triple_file)

        str_is_shuffle = "shuffle and split" if shuffle_data_split else "split without shuffle"
        if logger is not None:
            logger.debug("{} {} for {:.1f} training, {:.1f} validation and {:.1f} testing!".format( str_is_shuffle, triple_file, train_ratio, 1-train_ratio-test_ratio, test_ratio ))
            
        train_list, valid_list, test_list, e_map, r_map = processKG(kg_path, rel_vocab=rel_vocab, hop=hop, train_ratio=train_ratio, test_ratio=test_ratio, is_shuffle=shuffle_data_split, is_filter=filter_unseen_samples)

        # save ent_dic, rel_dic
        with open(e_map_file, 'w') as fout:
            for uri in e_map:
                fout.write('{}\t{}\n'.format(e_map[uri], uri))
        with open(r_map_file, 'w') as fout:
            for uri in r_map:
                fout.write('{}\t{}\n'.format(r_map[uri], uri))
        with open(train_file, 'w') as fout:
            for triple in train_list:
                fout.write('{}\t{}\t{}\n'.format(triple.h, triple.t, triple.r))
        with open(test_file, 'w') as fout:
            for triple in test_list:
                fout.write('{}\t{}\t{}\n'.format(triple.h, triple.t, triple.r))
        
        if len(valid_list) > 0:
            with open(valid_file, 'w') as fout:
                for triple in valid_list:
                    fout.write('{}\t{}\t{}\n'.format(triple.h, triple.t, triple.r))

    train_total, train_list = loadTriples(train_file)
    test_total, test_list = loadTriples(test_file)
    
    valid_total = 0
    valid_list = []
    if valid_file is not None and os.path.exists(valid_file):
        valid_total, valid_list = loadTriples(valid_file)
    
    if logger is not None:
        logger.info("Totally {} train, {} test and {} valid ratings!".format(train_total, test_total, valid_total))
    
    # get entity total
    entity_total, e_map = loadVocab(e_map_file)
    # get relation total
    relation_total, r_map = loadVocab(r_map_file)

    if logger is not None:
        logger.info("successfully load {} entities and {} relations!".format(entity_total, relation_total))

    train_iter = MakeTrainIterator(train_list, batch_size, negtive_samples=1)

    allHeadDict = {}
    allTailDict = {}
    if filter_wrong_corrupted:
        allTripleList = train_list + valid_list + test_list
        allHeadDict, allTailDict = buildAllDict(allTripleList)

    test_head_iter, test_tail_iter = getEvalIter(test_list, entity_total, batch_size, allHeadDict=allHeadDict, allTailDict=allTailDict)
    
    valid_head_iter = None
    valid_tail_iter = None
    if valid_total > 0:
        valid_head_iter, valid_tail_iter = getEvalIter(valid_list, entity_total, batch_size, allHeadDict=allHeadDict, allTailDict=allTailDict)

    datasets = (train_list, test_list, valid_list, allHeadDict, allTailDict)
    triple_iters = (train_iter, test_head_iter, test_tail_iter, valid_head_iter, valid_tail_iter)
    
    return datasets, triple_iters, e_map, r_map, entity_total, relation_total

if __name__ == "__main__":
    # Demo:
    data_path = "/Users/caoyixin/Github/joint-kg-recommender/datasets/ml1m/"

    # i2kg_file = os.path.join(data_path, 'i2kg_map.tsv')
    # i2kg_pairs = loadR2KgMap(i2kg_file)
    # e_set = set([p[1] for p in i2kg_pairs])

    rel_file = os.path.join(data_path+'kg/', 'relation_filter.dat')
    rel_vocab = set()
    with open(rel_file, 'r') as fin:
        for line in fin:
            rel_vocab.add(line.strip())

    _, triple_datasets = load_data(data_path, rel_vocab=rel_vocab)

    trainList, testList, validList, e_map, r_map, entity_total, relation_total = triple_datasets
    print("entity:{}, relation:{}!".format(entity_total, relation_total))
    print("totally triples for {} train, {} valid, and {} test!".format(len(trainList), len(validList), len(testList)))