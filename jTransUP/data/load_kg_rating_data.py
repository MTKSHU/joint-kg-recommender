import os
from jTransUP.data import load_rating_data, load_triple_data

def loadR2KgMap(filename):
    i2kg_pairs = []
    with open(filename, 'r') as fin:
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

def load_data(data_path, item_vocab=None, ent_vocab=None, rel_vocab=None, logger=None):
    
    _, triple_datasets, _, _ = load_triple_data.load_data(data_path, rel_vocab=rel_vocab, logger=logger)
    
    rating_datasets, _, _, _ = load_rating_data.load_data(data_path, item_vocab=item_vocab, logger=logger)

    return rating_datasets, triple_datasets, None, None