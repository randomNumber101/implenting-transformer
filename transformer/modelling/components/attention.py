import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from .basefunctions import softmax, linear, clones


def attention(query, key, value, mask=None, return_attn=False):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    print(f"Scores: {scores.size()}")
    if mask is not None:
        if mask.dim() == 2:  # Initial mask shape is [batch_size, seq_len]
            mask = mask.unsqueeze(1)  # Shape becomes [batch_size, 1, seq_len]
        mask = mask.expand(-1, query.size(1), key.size(1))  # Expand to [batch_size, num_queries, num_keys]
        print(f"Edited mask:  {mask.size()}")
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    print(f"Attn Matrix: {p_attn.size()}")
    result = torch.matmul(p_attn, value)
    print(f"Result: {result.size()}")
    return (result, p_attn) if return_attn else result


class Attention(nn.Module):
    def __init__(self, mask_future=False):
        super(Attention, self).__init__()
        self.mask_future = mask_future

    def forward(self, query, key, value, mask=None):
        print(
            f"Q: {query.size()} K: {key.size()} V: {value.size()} Mask: {mask.size() if mask is not None else 'None'}")
        if self.mask_future:
            seq_len = query.size(1)
            tri_matr = torch.tril(torch.ones((seq_len, seq_len), device=query.device), diagonal=0)
            future_mask = tri_matr.unsqueeze(0).expand(query.size(0), -1, -1)  # expand to batch size
            if mask is not None:
                mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
                mask = mask * future_mask
            else:
                mask = future_mask
        return attention(query, key, value, mask)

