import os
from jTransUP.data import load_rating_data, load_triple_data
from copy import deepcopy

def loadItemVocab(filename):
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

def loadEntityVocab(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
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

def loadR2KgMap(filename):
    i2kg_pairs = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 3 : continue
            i_id = int(line_split[0])
            kg_uri = line_split[2]
            i2kg_pairs.append( (i_id, kg_uri) )
    print("successful load {} item and entity pairs!".format(len(i2kg_pairs)))
    return i2kg_pairs

def remap(i2kg_pairs, i_map, e_map):
    i2kg_map = {}
    kg2i_map = {}
    for p in i2kg_pairs:
        mapped_i = i_map[p[0]]
        mapped_e = e_map[p[1]]
        i2kg_map[mapped_i] = mapped_e
        kg2i_map[mapped_e] = mapped_i
    return i2kg_map, kg2i_map

# map: org:id
# link: org(map1):org(map2)
def rebuildEntityItemVocab(map1, map2, links):
    new_map = {}
    index = 0
    has_map2 = set()
    remap1 = {}
    for org_id1 in map1:
        org_id2 = -1
        if org_id1 in links:
            org_id2 = links[org_id1]
            has_map2.add(org_id2)
        new_map[index] = (org_id1, org_id2)
        
        remap1[map1[org_id1]] = index
        index += 1

    remap2 = {}
    org_id1 = -1
    for org_id2 in map2:
        if org_id2 in has_map2 : continue
        new_map[index] = (org_id1, org_id2)

        remap2[map2[org_id2]] = index
        index += 1
    return new_map, remap1, remap2
            

def load_data(FLAGS, has_valid=True, logger=None, hop=1):
    data_path = os.path.join(FLAGS.data_path, FLAGS.dataset)
    kg_path = os.path.join(data_path, 'kg')

    i_map_file = os.path.join(data_path, "i_map.dat")
    e_map_file = os.path.join(kg_path, "e_map.dat")
    # get item total
    item_total, i_map = loadItemVocab(i_map_file)
    # get entity total
    entity_total, e_map = loadEntityVocab(e_map_file)

    # load mapped item vocab for filtering
    org_links = {}
    i2kg_file = os.path.join(data_path, 'i2kg_map.tsv')
    i2kg_pairs = loadR2KgMap(i2kg_file)
    for p in i2kg_pairs:
        org_links[p[0]] = p[1]

    new_map = None
    i_remap = None
    e_remap = None
    if FLAGS.share_embeddings:
        new_map, e_remap, i_remap = rebuildEntityItemVocab(e_map, i_map, org_links)

    rating_datasets, rating_iters, u_map, i_map, user_total, item_total = load_rating_data.load_data(data_path, FLAGS.batch_size, filter_wrong_corrupted=FLAGS.filter_wrong_corrupted, item_vocab=org_links if FLAGS.mapped_vocab_to_filter else None, logger=logger, filter_unseen_samples=FLAGS.filter_unseen_samples, shuffle_data_split=FLAGS.shuffle_data_split, train_ratio=FLAGS.train_ratio, test_ratio=FLAGS.test_ratio, i_remap=i_remap)

    rel_vocab = None
    if FLAGS.filter_relation:
        rel_vocab = set()
        rel_file = os.path.join(kg_path, 'relation_filter.dat')
        with open(rel_file, 'r') as fin:
            for line in fin:
                rel_vocab.add(line.strip())

    triple_datasets, triple_iters, e_map, r_map, entity_total, relation_total = load_triple_data.load_data(kg_path, FLAGS.batch_size, filter_wrong_corrupted=FLAGS.filter_wrong_corrupted, rel_vocab=rel_vocab, logger=logger, hop=hop, filter_unseen_samples=FLAGS.filter_unseen_samples, shuffle_data_split=FLAGS.shuffle_data_split, train_ratio=FLAGS.train_ratio, test_ratio=FLAGS.test_ratio, e_remap=e_remap)

    return rating_datasets, rating_iters, u_map, i_map, user_total, item_total, triple_datasets, triple_iters, e_map, r_map, entity_total, relation_total, new_map, e_remap, i_remap