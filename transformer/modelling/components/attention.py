import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from .basefunctions import softmax, linear, clones


def attention(query, key, value, mask=None, return_attn=False):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    result = torch.matmul(p_attn, value)
    return (result, p_attn) if return_attn else result


class Attention(nn.Module):
    def __init__(self, mask_future=False):
        super(Attention, self).__init__()
        self.mask_future = mask_future

    def forward(self, query, key, value, mask=None):
        if self.mask_future:
            seq_len_q = query.size(-2)
            seq_len_k = key.size(-2)
            # Create future mask
            future_mask = torch.tril(torch.ones((seq_len_q, seq_len_k), device=query.device)).bool()
            future_mask = future_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len_q, seq_len_k)
            future_mask = future_mask.expand(query.size(0), query.size(1), -1, -1)  # Expand to match batch and heads

            if mask is not None:
                # Combine existing mask with future mask
                mask = mask & future_mask
            else:
                mask = future_mask
        return attention(query, key, value, mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_heads, dim_keys=None, dim_values=None, mask_future=False):
        super().__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_keys = dim_keys if dim_keys is not None else dim_model // num_heads
        self.dim_values = dim_values if dim_values is not None else dim_model // num_heads

        # Shared linear projections
        self.query_transform = nn.Linear(dim_model, dim_model, bias=False)
        self.key_transform = nn.Linear(dim_model, dim_model, bias=False)
        self.value_transform = nn.Linear(dim_model, dim_model, bias=False)
        self.output_transform = nn.Linear(dim_model, dim_model, bias=False)

        # Attention mechanism
        self.attention = Attention(mask_future)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        query = self.query_transform(query).view(batch_size, seq_len_q, self.num_heads, self.dim_keys).transpose(1, 2)
        key = self.key_transform(key).view(batch_size, seq_len_k, self.num_heads, self.dim_keys).transpose(1, 2)
        value = self.value_transform(value).view(batch_size, seq_len_k, self.num_heads, self.dim_values).transpose(1, 2)

        if mask is not None:
            # If (batch_size, seq_len_k)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len_k)
                mask = mask.expand(-1, self.num_heads, seq_len_q,
                                   -1)  # Expand to (batch_size, num_heads, seq_len_q, seq_len_k)
            # If (batch_size, seq_len_q, seq_len_k)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            else:
                raise ValueError("Mask has an unexpected number of dimensions")

            mask = mask.bool()
        else:
            mask = None

        # Apply attention
        x = self.attention(query, key, value, mask=mask)

        # Concatenate heads and project out
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.dim_model)
        return self.output_transform(x)
