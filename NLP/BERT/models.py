from activation import gelu

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, vocab_size, max_seq_len, segments_num, embedding_dim, dropout_hidden_ratio):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.segment_embedding = nn.Embedding(segments_num, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        self.layerNorm = LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_hidden_ratio)

        self.max_seq_len = max_seq_len

    def forward(self, x, segments=None):
        positions = torch.arrange(self.max_seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand_as(x) # (S,) -> (B, S)
        
        if segments is None:
            segments = torch.zeros_like(x)

        embeddings = self.token_embedding(x) + self.segment_embedding(segments) + self.position_embedding(x)
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
        self.each_head_attention_size = embedding_dim // self.heads_num

    def transpose_for_scores(self, x):
        #split embedding to heads_num for seperate training
        new_x_shape = x.size()[:-1] + (self.heads_num, x.size(), self.each_head_attention_size)
        x = x.view(*new_x_shape)

        # convert (batch, sentence_length, head, each_head_attention) -> (batch, head, sentence_length, each_head_attention)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, x, attention_mask=None):
        query = self.query_embedding(x)
        key = self.key_embedding(x)
        value = self.value_embedding(x)

        query_layer = self.transpose_for_scores(x)
        key_layer = self.transpose_for_scores(x)
        value_layer = self.transpose_for_scores(x)

        #calculate attention score using query & key
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / match.sqrt(self.each_head_attention_size)

        if attention_mask is not None:
            #convert (batch, sentence_length) -> (batch, 1, 1, sentence_length) to mapping permuted batch masking
            attention_mask = attention_mask[:, None, None, :].float()
            attention_scores -= 10000.0 * (1.0 - attention_mask)

        #apply softmax & dropout per each head context embeddings
        attention_probs = self.dropout(F.softmax(attention_mask, dim=-1))           
        
        context_layer = torch.matmul(attention_probs, value_layer)

        # convert original shape for concatenation
        # (batch, head, sentence_length, each_head_attention) -> (batch, sentence_length, head, each_head_attention)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(*context_layer.size()[:-2], -1) # concatenate all attentions

        return context_layer

class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, feedFoward_dim):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, feedForward_dim)
        self.fc2 = nn.Linear(feedForward_dim, cfg.dim)

    def forward(self, x):
        # (Batch, Sentence_length, embedding) -> (Batch, Sentence_length, feedForward_embedding) -> (Batch, Sentenec_length, embedding)
        return self.fc2(gelu(self.fc1(x)))

class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, dropout_attention_ratio, heads_num, feedForward_dim, dropout_hidden_ratio):
        super().__init__()
        self.attention = ScaledDotProductAttention(embedding_dim, dropout_attention_ratio, heads_num)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.norm1 = LayerNorm(embedding_dim)

        self.posistionWiseFeedForward = PositionWiseFeedForward(feedForward_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_hidden_ratio)

    def forward(self, x, mask):
        attention = self.attention(x, mask)
        attention = self.norm1(x + self.dropout(self.projection(attention))) #residual connection
        attention = self.norm2(attention + self.dropout(self.positionWiseFeedForward(attention))) #residual connection

        return attention

class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, segments_num, embedding_dim, dropout_hidden_ratio, 
                blocks_num, dropout_attention_ratio, heads_num, feedForward_dim):
        super().__init__()
        self.embeddings = InputEmbedding(vocab_size, max_seq_len, segments_num, embedding_dim, dropout_hidden_ratio)
        self.encoderBlocks = nn.ModuleList([EncoderBlock(embedding_dim, dropout_attention_ratio, heads_num, feedForward_dim, dropout_hidden_ratio)
                                            for _ in range(blocks_num)])

    def forward(self, x, segment, mask):
        attention = self.embeddings(x, segments)
        for block in self.encoderBlocks:
            attention = block(attention, mask)

        return attention







