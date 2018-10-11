from jTransUP.utils.misc import evalProcess
import numpy as np

preds = [(1, np.array([0.81322029, 0.65334909, 0.44771267, 0.78720596, 0.53740775,
       0.59856147, 0.63714567, 0.5749356 , 0.79316778, 0.82273785])), (2, np.random.rand(10)), (4, np.random.rand(10))]
gold_dict = {1:set([1,3,4]), 2:set([1,3,4]), 3:set([1,3,4])}

for pred in preds:
    print(pred)

print(evalProcess(preds, gold_dict, descending=True, topn=5, num_processes=1))