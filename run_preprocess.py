from jTransUP.data.preprocess import processKG

kg_path = '/Users/caoyixin/Github/joint-kg-recommender/datasets/dbbook2014/kg/'
ent_vocab_file = kg_path + 'entity_vocab.dat'
rel_vocab_file = kg_path + 'predicate_vocab.dat'

processKG(kg_path, ent_vocab_file=ent_vocab_file, rel_vocab_file=rel_vocab_file)