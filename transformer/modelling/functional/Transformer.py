from collections import OrderedDict

import torch
import torch.nn as nn

from transformer.modelling.components.attention import MultiHeadAttention
from transformer.modelling.components.embedding import CombinedEmbedding
from transformer.modelling.functional.TransformerDecoder import TransformerDecoderLayer
from transformer.modelling.preprocessor.tokenizer import BPETokenizer

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


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout, max_len, device="cpu"):
        super().__init__()

        # Embedding and Positional Encoding
        self.embedding = CombinedEmbedding(vocab_size, d_model, max_len)

        # Encoder Stack
        self.encoder_layers = nn.ModuleList([
            BaseTransformerLayer(input_dim=d_model, num_heads=n_heads, feature_dim=dim_feedforward, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder Stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(input_dim=d_model, num_heads=n_heads, feature_dim=dim_feedforward, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output Projection
        self.output_layer = nn.Linear(d_model, vocab_size)

        self.device = device

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embed and positional encode source
        src = self.embedding(src)
        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, attention_mask=src_mask)

        # Embed and decode target
        tgt = self.embedding(tgt)
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, memory, encoder_attention_mask=src_mask, attention_mask=tgt_mask)

        # Project to vocabulary
        return self.output_layer(output)

    def generate(self, src, tokenizer: BPETokenizer, max_len=50, src_mask=None):
        self.eval()
        start_token_id = tokenizer.gpt2_tokenizer.bos_token_id
        end_token_id = tokenizer.gpt2_tokenizer.eos_token_id

        # Encode source with source mask
        src = self.embedding(src)
        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, attention_mask=src_mask)  # Pass src_mask to the encoder layers

        # Initialize the target sequence with the start token
        tgt = torch.tensor([[start_token_id]], device=src.device)

        for _ in range(max_len):
            tgt_emb = self.embedding(tgt)  # Embed the target sequence (includes positional encoding)
            output = tgt_emb
            for layer in self.decoder_layers:
                output = layer(output, memory, encoder_attention_mask=src_mask)
            logits = self.output_layer(output[:, -1, :])  # Take the last token
            next_token = logits.argmax(dim=-1)  # Greedy decoding: pick the token with the highest probability

            # Append the predicted token to the target sequence
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)

            # Stop if end-of-sequence token is generated
            if next_token.item() == end_token_id:
                break

        return tgt.squeeze(0).tolist()

    def to(self, device):
        self.device = device
        return super().to(device)

