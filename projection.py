import torch

def projection_transH_pytorch(original, norm):
	return original - torch.sum(original * norm, dim=1, keepdim=True) * norm