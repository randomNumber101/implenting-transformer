import torch
import torch.nn as nn
import copy

def softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True))
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)

def linear(x, w, b):
    return x @ w + b


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
