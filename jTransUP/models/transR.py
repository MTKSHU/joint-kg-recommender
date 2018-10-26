import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

from jTransUP.utils.misc import to_gpu, projection_transR_pytorch

def build_model(FLAGS, user_total, item_total, entity_total, relation_total, i_map=None, e_map=None, new_map=None):
    model_cls = TransRModel
    return model_cls(
                L1_flag = FLAGS.L1_flag,
                embedding_size = FLAGS.embedding_size,
                ent_total = entity_total,
                rel_total = relation_total
    )

class TransRModel(nn.Module):
    def __init__(self,
                L1_flag,
                embedding_size,
                ent_total,
                rel_total
                ):
        super(TransRModel, self).__init__()
        self.L1_flag = L1_flag
        self.embedding_size = embedding_size
        self.ent_total = ent_total
        self.rel_total = rel_total
        self.is_pretrained = False
        self.max_entity_batch = 10

        ent_weight = torch.FloatTensor(self.ent_total, self.embedding_size)
        rel_weight = torch.FloatTensor(self.rel_total, self.embedding_size)
        proj_weight = torch.FloatTensor(self.rel_total, self.embedding_size * self.embedding_size)
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)
        
        if self.is_pretrained:
            nn.init.eye(proj_weight)
            proj_weight = proj_weight.view(-1).expand(self.relation_total, -1)
        else:
            nn.init.xavier_uniform(proj_weight)
            
        # init user and item embeddings
        self.ent_embeddings = nn.Embedding(self.ent_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.rel_total, self.embedding_size)
        self.proj_embeddings = nn.Embedding(self.rel_total, self.embedding_size * self.embedding_size)

        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.proj_embeddings.weight = nn.Parameter(proj_weight)

        normalize_ent_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_rel_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        # normalize_proj_emb = F.normalize(self.proj_embeddings.weight.data, p=2, dim=1)

        self.ent_embeddings.weight.data = normalize_ent_emb
        self.rel_embeddings.weight.data = normalize_rel_emb
        # self.proj_embeddings.weight.data = normalize_proj_emb

        self.ent_embeddings = to_gpu(self.ent_embeddings)
        self.rel_embeddings = to_gpu(self.rel_embeddings)
        self.proj_embeddings = to_gpu(self.proj_embeddings)

    def forward(self, h, t, r):
        h_e = self.ent_embeddings(h)
        t_e = self.ent_embeddings(t)
        r_e = self.rel_embeddings(r)
        proj_e = self.proj_embeddings(r)

        proj_h_e = projection_transR_pytorch(h_e, proj_e)
        proj_t_e = projection_transR_pytorch(t_e, proj_e)

        if self.L1_flag:
            score = torch.sum(torch.abs(proj_h_e + r_e - proj_t_e), 1)
        else:
            score = torch.sum((proj_h_e + r_e - proj_t_e) ** 2, 1)
        return score
    
    def evaluateHead(self, t, r):
        batch_size = len(t)
        # batch * dim
        t_e = self.ent_embeddings(t)
        r_e = self.rel_embeddings(r)
        # batch* dim*dim
        proj_e = self.proj_embeddings(r)
        # batch * dim
        proj_t_e = projection_transR_pytorch(t_e, proj_e)
        c_h_e = proj_t_e - r_e
        
        score = []
        # entity * dim
        for i, single_proj in enumerate(proj_e):
            single_ent_e = projection_transR_pytorch(self.ent_embeddings.weight, single_proj)
            che_expand = c_h_e[i].expand(self.ent_total, self.embedding_size)
            # entity
            if self.L1_flag:
                single_score = torch.sum(torch.abs(che_expand-single_ent_e), 1)
            else:
                single_score = torch.sum((che_expand-single_ent_e) ** 2, 1)
            score.append(single_score)

        return torch.cat(score, dim=0)
    
    def evaluateTail(self, h, r):
        batch_size = len(h)
        # batch * dim
        h_e = self.ent_embeddings(h)
        r_e = self.rel_embeddings(r)
        # batch* dim*dim
        proj_e = self.proj_embeddings(r)
        # batch * dim
        proj_h_e = projection_transR_pytorch(h_e, proj_e)
        c_t_e = proj_h_e + r_e
        
        score = []
        # entity * dim
        for i, single_proj in enumerate(proj_e):
            single_ent_e = projection_transR_pytorch(self.ent_embeddings.weight, single_proj)
            cte_expand = c_t_e[i].expand(self.ent_total, self.embedding_size)
            # entity
            if self.L1_flag:
                single_score = torch.sum(torch.abs(cte_expand-single_ent_e), 1)
            else:
                single_score = torch.sum((cte_expand-single_ent_e) ** 2, 1)
            score.append(single_score)
        return torch.cat(score, dim=0)