import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

from jTransUP.utils.misc import to_gpu

def build_model(FLAGS, user_total, item_total):
    model_cls = BPRMF
    return model_cls(
                FLAGS.embedding_size,
                user_total,
                item_total)

class BPRMF(nn.Module):
    def __init__(self,
                embedding_size,
                user_total,
                item_total,
                ):
        super(BPRMF, self).__init__()
        self.embedding_size = embedding_size
        self.user_total = user_total
        self.item_total = item_total

        # init user and item embeddings
        self.user_embeddings = to_gpu(nn.Embedding(self.user_total, self.embedding_size))
        self.item_embeddings = to_gpu(nn.Embedding(self.item_total, self.embedding_size))
        self.user_embeddings.weight.requires_grad = True
        self.item_embeddings.weight.requires_grad = True


    def forward(self, u_ids, i_ids):
        u_e = self.user_embeddings(to_gpu(V(torch.LongTensor(u_ids))))
        i_e = self.item_embeddings(to_gpu(V(torch.LongTensor(i_ids))))
        return torch.bmm(u_e.unsqueeze(1), i_e.unsqueeze(2)).squeeze()