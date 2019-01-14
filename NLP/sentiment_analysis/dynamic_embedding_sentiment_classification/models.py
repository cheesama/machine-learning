import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta  = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Dynamic_embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim) # token embedding
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim) # position embedding
        #self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim) # segment(token type) embedding

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.dropout)

     def forward(self, x, seg=None):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)

        #e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        e = self.tok_embed(x) + self.pos_embed(pos)
        
        return self.drop(self.norm(e))

class WordCNN(nn.Module):
    def __init__(self, vocab_size, config):
        #considering resnet18, hidden size is 224
        self.vocab_size = vocab_size
        self.embedding = Dynamic_embeddings(config)

        self.conv = nn.ModuleList([
            nn.Conv2d(
                in_channels=2,
                out_channels=config.n_channel_per_window,
                kernel_size=(3, config.hidden_size)),

            nn.Conv2d(
                in_channels=2,
                out_channels=config.n_channel_per_window,
                kernel_size=(4, config.hidden_size)),

            nn.Conv2d(
                in_channels=2,
                out_channels=config.n_channel_per_window,
                kernel_size=(5, config.hidden_size))
        ])

        n_total_channels = len(self.conv) * config.n_channel_per_window

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(n_total_channels, config.label_size)

    def forward(self, x):
        """
        Args:
            x: [batch_size, max_seq_len]
        Return:
            logit: [batch_size, label_size]
        """

        # [batch_size, max_seq_len, hidden_size]
        x = self.embedding(x)

        # [batch_size, 1, max_seq_len, hidden_size]
        x = x.unsqueeze(1)

        # Apply Convolution filter followed by Max-pool
        out_list = []

        for conv in self.conv:
            ########## Convolution #########

            # [batch_size, n_kernels, _, 1]
            x_ = F.relu(conv(x))

            # [batch_size, n_kernels, _]
            x_ = x_.squeeze(3)

            ########## Max-pool #########

            # [batch_size, n_kernels, 1]
            x_ = F.max_pool1d(x_, x_.size(2))

            # [batch_size, n_kernels]
            x_ = x_.squeeze(2)

            out_list.append(x_)

        # [batch_size, 3 x n_kernels]
        out = torch.cat(out_list, 1)

        ######## Dropout ########
        out = self.dropout(out)

        # [batch_size, label_size]
        logit = self.fc(out)

        return logit

