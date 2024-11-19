import torch
import torch.nn as nn
from collections import OrderedDict

from transformer.modelling.components.attention import MultiHeadAttention


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads=2, feature_dim=6, dropout=0.0):
        super().__init__()

        # Self-attention layer with future masking
        self.self_attention = MultiHeadAttention(dim_model=input_dim, num_heads=num_heads, mask_future=True)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(input_dim)

        # Encoder cross-attention layer
        self.encoder_attention = MultiHeadAttention(dim_model=input_dim, num_heads=num_heads, mask_future=False)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(input_dim)

        # Position-wise feedforward network
        self.feature_transformation = nn.Sequential(
            OrderedDict([
                ("linear1", nn.Linear(input_dim, feature_dim)),
                ("activation", nn.ReLU()),
                ("linear2", nn.Linear(feature_dim, input_dim))
            ])
        )
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm_3 = nn.LayerNorm(input_dim)

    def forward(self, x, encoder_output, encoder_attention_mask=None, attention_mask=None):
        # Self-attention with future masking
        x_attn = self.self_attention(x, x, x, mask=attention_mask)
        x = x + self.dropout1(x_attn)
        x = self.layer_norm_1(x)

        # Encoder cross-attention
        x_cross_attn = self.encoder_attention(x, encoder_output, encoder_output, mask=encoder_attention_mask)
        x = x + self.dropout2(x_cross_attn)
        x = self.layer_norm_2(x)

        # Position-wise feedforward
        x_ff = self.feature_transformation(x)
        x = x + self.dropout3(x_ff)
        x = self.layer_norm_3(x)

        return x
