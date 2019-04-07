import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1m keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)

        return self.gamma * x + self.beta


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, max_seq_len, segments_num, embedding_dim, dropout_input_ratio):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.segment_embedding = nn.Embedding(segments_num, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        self.layerNorm = LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_input_ratio)

        self.max_seq_len = max_seq_len

    def forward(self, x):
        positions = torch.arrange(self.max_seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand_as(x) # (S,) -> (B, S)

        embeddings = self.token_embedding(x) + self.segment_embedding(x) + self.position_embedding(x)
        embeddings = self.layerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embedding_dim, dropout_attention_ratio, heads_num):
        self.query_embedding = nn.Linear(embedding_dim, embedding_dim)
        self.key_embedding = nn.Linear(embedding_dim, embedding_dim)
        self.value_embedding = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout_attention_ratio)
        self.heads_num = heads_num

    def forward(self, x, mask):
        query = self.query_embedding(x)
        key = self.key_embedding(x)
        value = self.value_embedding(x)


