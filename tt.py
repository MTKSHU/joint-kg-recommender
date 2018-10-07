from jTransUP.utils.misc import getPerformance

pred_dict = {'a':[(1,0.1),(2,0.2),(3,0.3)], 'b':[(4,0.4),(5,0.5)]}
gold_dict = {'a':[2,3], 'b':[5,4]}

f1, p, r, hit, ndcg, mean_rank = getPerformance(pred_dict, gold_dict)

print("f1:{:.4f},p:{:.4f},r:{:.4f},hit:{:.4f},ndcg:{:.4f},mean rank:{:.4f}!".format(f1, p, r, hit, ndcg, mean_rank))