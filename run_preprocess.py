from jTransUP.data.load_rating_data import load_data
from jTransUP.data.load_kg_rating_data import loadR2KgMap
import os

data_path = "/Users/caoyixin/Github/joint-kg-recommender/datasets/dbbook2014/"
# data_path = "/Users/caoyixin/Github/joint-kg-recommender/datasets/ml1m/"

batch_size = 10
from jTransUP.data.load_kg_rating_data import loadR2KgMap

i2kg_file = os.path.join(data_path, 'i2kg_map.tsv')
i2kg_pairs = loadR2KgMap(i2kg_file)
i_set = set([p[0] for p in i2kg_pairs])

datasets, rating_iters, u_map, i_map, user_total, item_total = load_data(data_path, batch_size, item_vocab=i_set)

trainList, testDict, validDict, allDict, testTotal, validTotal = datasets
print("user:{}, item:{}!".format(user_total, item_total))
print("totally ratings for {} train, {} valid, and {} test!".format(len(trainList), validTotal, testTotal))