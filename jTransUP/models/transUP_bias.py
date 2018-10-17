import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

from jTransUP.utils.misc import to_gpu, projection_transH_pytorch

def build_model(FLAGS, user_total, item_total, entity_total, relation_total):
    model_cls = BTransUPModel
    return model_cls(
                L1_flag = FLAGS.L1_flag,
                embedding_size = FLAGS.embedding_size,
                user_total = user_total,
                item_total = item_total,
                preference_total = FLAGS.num_preferences
    )

class BTransUPModel(nn.Module):
    def __init__(self,
                L1_flag,
                embedding_size,
                user_total,
                item_total,
                preference_total
                ):
        super(BTransUPModel, self).__init__()
        self.L1_flag = L1_flag
        self.embedding_size = embedding_size
        self.user_total = user_total
        self.item_total = item_total
        self.preference_total = preference_total
        self.is_pretrained = False

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
        self.pref_embeddings = nn.Embedding(self.preference_total, self.embedding_size)
        self.norm_embeddings = nn.Embedding(self.preference_total, self.embedding_size)
        self.pref_embeddings.weight = nn.Parameter(pref_weight)
        self.norm_embeddings.weight = nn.Parameter(norm_weight)
        normalize_pref_emb = F.normalize(self.pref_embeddings.weight.data, p=2, dim=1)
        normalize_norm_emb = F.normalize(self.norm_embeddings.weight.data, p=2, dim=1)
        self.pref_embeddings.weight.data = normalize_pref_emb
        self.norm_embeddings.weight.data = normalize_norm_emb

        self.user_embeddings = to_gpu(self.user_embeddings)
        self.item_embeddings = to_gpu(self.item_embeddings)
        self.pref_embeddings = to_gpu(self.pref_embeddings)
        self.norm_embeddings = to_gpu(self.norm_embeddings)

        user_bias = torch.FloatTensor(self.user_total)
        item_bias = torch.FloatTensor(self.item_total)
        nn.init.constant(user_bias, 0)
        nn.init.constant(item_bias, 0)
        self.user_bias = nn.Embedding(self.user_total, 1)
        self.item_bias = nn.Embedding(self.item_total, 1)
        self.user_bias.weight = nn.Parameter(user_bias, 1)
        self.item_bias.weight = nn.Parameter(item_bias, 1)

        self.user_bias = to_gpu(self.user_bias)
        self.item_bias = to_gpu(self.item_bias)

    def forward(self, u_ids, i_ids):
        u_e = self.user_embeddings(u_ids)
        i_e = self.item_embeddings(i_ids)
        # use item and user embedding to compute preference distribution
        pre_probs = torch.matmul(u_e + i_e, torch.t(self.pref_embeddings.weight)) / 2
        r_e = torch.matmul(pre_probs, self.pref_embeddings.weight)
        norm = torch.matmul(pre_probs, self.norm_embeddings.weight)

        proj_u_e = projection_transH_pytorch(u_e, norm)
        proj_i_e = projection_transH_pytorch(i_e, norm)

        i_b = self.item_bias(i_ids).squeeze()

        if self.L1_flag:
            score = torch.abs(i_b - torch.sum(torch.abs(proj_u_e + r_e - proj_i_e), 1))
        else:
            score = torch.abs(i_b - torch.sum((proj_u_e + r_e - proj_i_e) ** 2, 1))
        return score
    
    def evaluate(self, u_ids):
        batch_size = len(u_ids)
        u = self.user_embeddings(u_ids)
        # expand u and i to pair wise match, batch * item * dim 
        u_e = u.expand(self.item_total, batch_size, self.embedding_size).permute(1, 0, 2)
        i_e = self.item_embeddings.weight.expand(batch_size, self.item_total, self.embedding_size)

        # batch * item * pref
        pre_probs = torch.matmul(u_e + i_e, torch.t(self.pref_embeddings.weight)) / 2
        # batch * item * dim
        r_e = torch.matmul(pre_probs, self.pref_embeddings.weight)
        norm = torch.matmul(pre_probs, self.norm_embeddings.weight)

        # batch * item * dim
        proj_u_e = projection_transH_pytorch(u_e, norm)
        proj_i_e = projection_transH_pytorch(i_e, norm)

        i_b = self.item_bias.weight.squeeze().expand(batch_size, self.item_total)

        # batch * item
        if self.L1_flag:
            score = torch.abs(i_b - torch.sum(torch.abs(proj_u_e + r_e - proj_i_e), 2))
        else:
            score = torch.abs(i_b - torch.sum((proj_u_e + r_e - proj_i_e) ** 2, 2))
        return score