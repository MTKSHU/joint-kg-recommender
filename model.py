import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from projection import *

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor

class TransUPModel(nn.Module):
	def __init__(self, config):
		super(TransUPModel, self).__init__()
		self.learning_rate = config.learning_rate
		self.early_stopping_round = config.early_stopping_round
		self.L1_flag = config.L1_flag
		self.filter = config.filter
		self.embedding_size = config.embedding_size
		self.user_total = config.user_total
		self.item_total = config.item_total
		self.batch_size = config.batch_size
		self.preference_total = config.preference_total

		user_weight = floatTensor(self.user_total, self.embedding_size)
		item_weight = floatTensor(self.item_total, self.embedding_size)
		pref_weight = floatTensor(self.preference_total, self.embedding_size)
		norm_weight = floatTensor(self.preference_total, self.embedding_size)
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

	def forward(self, pos_u, pos_i, neg_u, neg_i):
		pos_u_e = self.user_embeddings(pos_u)
		pos_i_e = self.item_embeddings(pos_i)
		# use item and user embedding to compute preference distribution
		pre_probs = torch.matmul(pos_u_e + pos_i_e, torch.t(self.pref_weight)) / 2
		pos_r_e = torch.matmul(pre_probs, self.pref_weight)
		pos_norm = torch.matmul(pre_probs, self.norm_weight)

		neg_u_e = self.user_embeddings(neg_u)
		neg_i_e = self.item_embeddings(neg_i)
		# use item and user embedding to compute preference distribution
		pre_probs = torch.matmul(neg_u_e + neg_i_e, torch.t(self.pref_weight)) / 2
		
		neg_r_e = torch.matmul(pre_probs, self.pref_weight)
		neg_norm = torch.matmul(pre_probs, self.norm_weight)

		pos_u_e = projection_transH_pytorch(pos_u_e, pos_norm)
		pos_i_e = projection_transH_pytorch(pos_i_e, pos_norm)
		neg_u_e = projection_transH_pytorch(neg_u_e, neg_norm)
		neg_i_e = projection_transH_pytorch(neg_i_e, neg_norm)

		if self.L1_flag:
			pos = torch.sum(torch.abs(pos_u_e + pos_r_e - pos_i_e), 1)
			neg = torch.sum(torch.abs(neg_u_e + neg_r_e - neg_i_e), 1)
		else:
			pos = torch.sum((pos_u_e + pos_r_e - pos_i_e) ** 2, 1)
			neg = torch.sum((neg_u_e + neg_r_e - neg_i_e) ** 2, 1)
		return pos, neg, self.pref_weight, self.norm_weight