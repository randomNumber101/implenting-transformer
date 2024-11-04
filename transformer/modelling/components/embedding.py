import torch
import math
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute position-dependent values for sine and cosine functions
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register positional encodings as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encodings to the input embeddings
        x = x + self.pe[:x.size(1), :]
        return x


class CombinedEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len=5000):
        super(CombinedEmbedding, self).__init__()
        # Initialize word embeddings
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Initialize positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)

    def forward(self, x):
        # Get word embeddings
        word_embeddings = self.word_embedding(x)
        # Add positional encodings
        embeddings = self.positional_encoding(word_embeddings)
        return embeddings
