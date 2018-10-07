import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from jTransUP.utils.misc import to_gpu

def orthogonalLoss(rel_embeddings, norm_embeddings):
	return torch.sum(torch.sum(norm_embeddings * rel_embeddings, dim=1, keepdim=True) ** 2 / torch.sum(rel_embeddings ** 2, dim=1, keepdim=True))

def normLoss(embeddings, dim=1):
	norm = torch.sum(embeddings ** 2, dim=dim, keepdim=True)
	return torch.sum(torch.max(norm - 1.0, to_gpu(autograd.Variable(torch.FloatTensor([0.0])))))

def bprLoss(pos, neg, target=1.0):
	loss = - F.logsigmoid(target * ( pos - neg ))
	return loss.mean()