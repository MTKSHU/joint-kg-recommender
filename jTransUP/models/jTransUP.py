import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

from jTransUP.utils.misc import to_gpu, projection_transH_pytorch
from jTransUP.models.transH import TransHModel
from jTransUP.models.transUP import TransUPModel

def build_model(FLAGS, user_total, item_total, entity_total, relation_total, i_map=None, e_map=None, new_map=None):
    model_cls = jTransUPModel
    return model_cls(
                L1_flag = FLAGS.L1_flag,
                embedding_size = FLAGS.embedding_size,
                user_total = user_total,
                item_total = item_total,
                entity_total = entity_total,
                relation_total = relation_total,
                isShare = FLAGS.share_embeddings
    )

class jTransUPModel(nn.Module):
    def __init__(self,
                L1_flag,
                embedding_size,
                user_total,
                item_total,
                entity_total,
                relation_total,
                isShare
                ):
        super(jTransUPModel, self).__init__()
        self.L1_flag = L1_flag
        self.is_share = isShare
        self.embedding_size = embedding_size
        self.user_total = user_total
        self.item_total = item_total
        self.ent_total = entity_total
        self.rel_total = relation_total
        self.is_pretrained = False
        # transup
        user_weight = torch.FloatTensor(self.user_total, self.embedding_size)
        nn.init.xavier_uniform(user_weight)
        self.user_embeddings = nn.Embedding(self.user_total, self.embedding_size)
        self.user_embeddings.weight = nn.Parameter(user_weight)
        normalize_user_emb = F.normalize(self.user_embeddings.weight.data, p=2, dim=1)
        self.user_embeddings.weight.data = normalize_user_emb
        self.user_embeddings = to_gpu(self.user_embeddings)

        # transh
        
        rel_weight = torch.FloatTensor(self.rel_total, self.embedding_size)
        norm_weight = torch.FloatTensor(self.rel_total, self.embedding_size)
        nn.init.xavier_uniform(rel_weight)
        nn.init.xavier_uniform(norm_weight)
        self.rel_embeddings = nn.Embedding(self.rel_total, self.embedding_size)
        self.norm_embeddings = nn.Embedding(self.rel_total, self.embedding_size)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.norm_embeddings.weight = nn.Parameter(norm_weight)
        normalize_rel_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        normalize_norm_emb = F.normalize(self.norm_embeddings.weight.data, p=2, dim=1)
        self.rel_embeddings.weight.data = normalize_rel_emb
        self.norm_embeddings.weight.data = normalize_norm_emb
        self.rel_embeddings = to_gpu(self.rel_embeddings)
        self.norm_embeddings = to_gpu(self.norm_embeddings)

        # is share embedding
        ent_weight = torch.FloatTensor(self.ent_total, self.embedding_size)
        nn.init.xavier_uniform(ent_weight)
        self.ent_embeddings = nn.Embedding(self.ent_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        normalize_ent_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_ent_emb
        self.ent_embeddings = to_gpu(self.ent_embeddings)

        if self.is_share:
            assert self.item_total == self.ent_total, "item numbers didn't match entities!"
            self.item_embeddings = self.ent_embeddings
        else:
            item_weight = torch.FloatTensor(self.item_total, self.embedding_size)
            nn.init.xavier_uniform(item_weight)
            self.item_embeddings = nn.Embedding(self.item_total, self.embedding_size)
            self.item_embeddings.weight = nn.Parameter(item_weight)
            normalize_item_emb = F.normalize(self.item_embeddings.weight.data, p=2, dim=1)
            self.item_embeddings.weight.data = normalize_item_emb
            self.item_embeddings = to_gpu(self.item_embeddings)
        
        # to be consistent with transUP
        self.pref_embeddings = self.rel_embeddings

    def forward(self, ratings, triples, is_rec=True):
        
        if is_rec and ratings is not None:
            u_ids, i_ids = ratings
            u_e = self.user_embeddings(u_ids)
            i_e = self.item_embeddings(i_ids)
            # use item and user embedding to compute preference distribution
            pre_probs = torch.matmul(u_e + i_e, torch.t(self.rel_embeddings.weight)) / 2
            r_e = torch.matmul(pre_probs, self.rel_embeddings.weight)
            norm = torch.matmul(pre_probs, self.norm_embeddings.weight)

            proj_u_e = projection_transH_pytorch(u_e, norm)
            proj_i_e = projection_transH_pytorch(i_e, norm)

            if self.L1_flag:
                score = torch.sum(torch.abs(proj_u_e + r_e - proj_i_e), 1)
            else:
                score = torch.sum((proj_u_e + r_e - proj_i_e) ** 2, 1)
        elif not is_rec and triples is not None:
            h, t, r = triples
            h_e = self.ent_embeddings(h)
            t_e = self.ent_embeddings(t)
            r_e = self.rel_embeddings(r)
            norm_e = self.norm_embeddings(r)

            proj_h_e = projection_transH_pytorch(h_e, norm_e)
            proj_t_e = projection_transH_pytorch(t_e, norm_e)

            if self.L1_flag:
                score = torch.sum(torch.abs(proj_h_e + r_e - proj_t_e), 1)
            else:
                score = torch.sum((proj_h_e + r_e - proj_t_e) ** 2, 1)
        else:
            raise NotImplementedError
        
        return score
    
    def evaluateRec(self, u_ids, all_i_ids=None):
        batch_size = len(u_ids)
        all_i = self.item_embeddings(all_i_ids) if all_i_ids is not None and self.is_share else self.item_embeddings.weight
        item_total, dim = all_i.size()

        u = self.user_embeddings(u_ids)
        # expand u and i to pair wise match, batch * item * dim 
        u_e = u.expand(item_total, batch_size, dim).permute(1, 0, 2)
        
        i_e = all_i.expand(batch_size, item_total, dim)

        # batch * item * pref
        pre_probs = torch.matmul(u_e + i_e, torch.t(self.rel_embeddings.weight)) / 2
        # batch * item * dim
        r_e = torch.matmul(pre_probs, self.rel_embeddings.weight)
        norm = torch.matmul(pre_probs, self.norm_embeddings.weight)

        # batch * item * dim
        proj_u_e = projection_transH_pytorch(u_e, norm)
        proj_i_e = projection_transH_pytorch(i_e, norm)

        # batch * item
        if self.L1_flag:
            score = torch.sum(torch.abs(proj_u_e + r_e - proj_i_e), 2)
        else:
            score = torch.sum((proj_u_e + r_e - proj_i_e) ** 2, 2)
        return score
    
    def evaluateHead(self, t, r, all_e_ids=None):
        batch_size = len(t)
        all_e = self.ent_embeddings(all_e_ids) if all_e_ids is not None and self.is_share else self.ent_embeddings.weight
        ent_total, dim = all_e.size()
        # batch * dim
        t_e = self.ent_embeddings(t)
        r_e = self.rel_embeddings(r)
        norm_e = self.norm_embeddings(r)

        proj_t_e = projection_transH_pytorch(t_e, norm_e)
        c_h_e = proj_t_e - r_e
        
        # batch * entity * dim
        c_h_expand = c_h_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)

        # batch * entity * dim
        norm_expand = norm_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)

        ent_expand = all_e.expand(batch_size, ent_total, dim)
        proj_ent_e = projection_transH_pytorch(ent_expand, norm_expand)

        # batch * entity
        if self.L1_flag:
            score = torch.sum(torch.abs(c_h_expand-proj_ent_e), 2)
        else:
            score = torch.sum((c_h_expand-proj_ent_e) ** 2, 2)
        return score
    
    def evaluateTail(self, h, r, all_e_ids=None):
        batch_size = len(h)
        all_e = self.ent_embeddings(all_e_ids) if all_e_ids is not None and self.is_share else self.ent_embeddings.weight
        ent_total, dim = all_e.size()
        # batch * dim
        h_e = self.ent_embeddings(h)
        r_e = self.rel_embeddings(r)
        norm_e = self.norm_embeddings(r)

        proj_h_e = projection_transH_pytorch(h_e, norm_e)
        c_t_e = proj_h_e + r_e
        
        # batch * entity * dim
        c_t_expand = c_t_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)

        # batch * entity * dim
        norm_expand = norm_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)
        
        ent_expand = all_e.expand(batch_size, ent_total, dim)
        proj_ent_e = projection_transH_pytorch(ent_expand, norm_expand)

        # batch * entity
        if self.L1_flag:
            score = torch.sum(torch.abs(c_t_expand-proj_ent_e), 2)
        else:
            score = torch.sum((c_t_expand-proj_ent_e) ** 2, 2)
        return score
    
    def getPreferences(self, u_id, i_ids):
        item_num = len(i_ids)
        # item_num * dim
        u_e = self.user_embeddings(u_id.expand(item_num))
        # item_num * dim
        i_e = self.item_embeddings(i_ids)
        # item_num * relation_total
        pre_probs = torch.matmul(u_e + i_e, torch.t(self.rel_embeddings.weight)) / 2

        return pre_probs