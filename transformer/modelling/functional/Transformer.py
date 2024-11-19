from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.modelling.components.attention import MultiHeadAttention

'''
Using nn.Layernorm nonetheless, as tests were conducted with nn.Layernorm and there is a small difference (<1e10)
'''


class LayerNorm(nn.Module):
    def __init__(self, dim_model, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim_model))
        self.bias = nn.Parameter(torch.zeros(dim_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.weight * out + self.bias
        return out


class BaseTransformerLayer(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, feature_dim: int, dropout: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.self_attention = MultiHeadAttention(dim_model=input_dim, num_heads=num_heads)
        self.layer_norm_1 = nn.LayerNorm(input_dim)

        self.feature_transformation = nn.Sequential(
            OrderedDict([
                ("linear1", nn.Linear(input_dim, feature_dim, bias=True)),
                ("activation", nn.ReLU()),
                ("dropout", nn.Dropout(dropout)),
                ("linear2", nn.Linear(feature_dim, input_dim, bias=True))
            ])
        )
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # Attention Layer
        x_attn = self.dropout1(self.self_attention(x, x, x, mask=attention_mask))
        x = self.layer_norm_1(x + x_attn)

        # FF layer

        x_ff = self.dropout2(self.feature_transformation(x))
        x = self.layer_norm_2(x + x_ff)
        return x
