import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

from jTransUP.utils.misc import to_gpu, projection_transH_pytorch

def build_model(FLAGS, user_total, item_total):
    model_cls = TransUPModel
    return model_cls(
                L1_flag = FLAGS.L1_flag,
                embedding_size = FLAGS.embedding_size,
                user_total = user_total,
                item_total = item_total,
                preference_total = FLAGS.num_preferences
    )

class TransUPModel(nn.Module):
    def __init__(self,
                L1_flag,
                embedding_size,
                user_total,
                item_total,
                preference_total
                ):
        super(TransUPModel, self).__init__()
        self.L1_flag = L1_flag
        self.embedding_size = embedding_size
        self.user_total = user_total
        self.item_total = item_total
        self.preference_total = preference_total

        user_weight = torch.FloatTensor(self.user_total, self.embedding_size)
        item_weight = torch.FloatTensor(self.item_total, self.embedding_size)
        pref_weight = torch.FloatTensor(self.preference_total, self.embedding_size)
        norm_weight = torch.FloatTensor(self.preference_total, self.embedding_size)
        nn.init.xavier_uniform(user_weight)
        nn.init.xavier_uniform(item_weight)
        nn.init.xavier_uniform(pref_weight)
        nn.init.xavier_uniform(norm_weight)
        # init user and item embeddings
        self.user_embeddings = nn.Embedding(self.user_total, self.embedding_size)
        self.item_embeddings = nn.Embedding(self.item_total, self.embedding_size)
        self.user_embeddings.weight = nn.Parameter(user_weight)
        self.item_embeddings.weight = nn.Parameter(item_weight)
        normalize_user_emb = F.normalize(self.user_embeddings.weight.data, p=2, dim=1)
        normalize_item_emb = F.normalize(self.item_embeddings.weight.data, p=2, dim=1)
        self.user_embeddings.weight.data = normalize_user_emb
        self.item_embeddings.weight.data = normalize_item_emb
        # init preference parameters
        self.pref_weight = nn.Parameter(pref_weight)
        self.norm_weight = nn.Parameter(norm_weight)
        normalize_pre_emb = F.normalize(self.pref_weight.data, p=2, dim=1)
        normalize_norm_emb = F.normalize(self.norm_weight.data, p=2, dim=1)
        self.pref_weight.data = normalize_pre_emb
        self.norm_weight.data = normalize_norm_emb

        self.user_embeddings = to_gpu(self.user_embeddings)
        self.item_embeddings = to_gpu(self.item_embeddings)
        self.pref_weight = self.pref_weight
        self.norm_weight = self.norm_weight

    def forward(self, u_ids, i_ids):
        u_e = self.user_embeddings(to_gpu(V(torch.LongTensor(u_ids))))
        i_e = self.item_embeddings(to_gpu(V(torch.LongTensor(i_ids))))
        # use item and user embedding to compute preference distribution
        pre_probs = torch.matmul(u_e + i_e, torch.t(self.pref_weight)) / 2
        r_e = torch.matmul(pre_probs, self.pref_weight)
        norm = torch.matmul(pre_probs, self.norm_weight)

        proj_u_e = projection_transH_pytorch(u_e, norm)
        proj_i_e = projection_transH_pytorch(i_e, norm)

        if self.L1_flag:
            score = torch.sum(torch.abs(proj_u_e + r_e - proj_i_e), 1)
        else:
            score = torch.sum((proj_u_e + r_e - proj_i_e) ** 2, 1)
        return score