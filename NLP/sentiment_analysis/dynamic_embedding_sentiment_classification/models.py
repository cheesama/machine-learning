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
        super(LayerNorm, self).__init__()
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
        super(Dynamic_embeddings, self).__init__()

        #for square input form, fit max_seq_len as dim size
        cfg.max_len = cfg.dim

        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim) # token embedding
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim) # position embedding
        #self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim) # segment(token type) embedding

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.dropout)
     
    def forward(self, x):
        seq_len = x.size()[1]
        pos_forward = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos_forward = pos_forward.unsqueeze(0).expand_as(x) # (S,) -> (B, S)
        pos_backward = torch.arange(seq_len, dtype=torch.long, device=x.device).flip(0)
        pos_backward = pos_backward.unsqueeze(0).expand_as(x) # (S,) -> (B, S)

        #generate mask embedding
        x_mask = self.tok_embed(x)
        masking_target_prob = 0.15
        masking_target_index = np.random.choice(range(x_mask.size()[0]), int(x_mask.size()[0] * masking_target_prob))

        for each_masking_index in masking_target_index:
            x_mask[each_masking_index] = 0   

        #e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        forward_embedding = self.tok_embed(x) + self.pos_embed(pos_forward)
        mask_embedding = x_mask
        backward_embedding = self.tok_embed(x) + self.pos_embed(pos_backward)

        e = torch.stack((forward_embedding, mask_embedding, backward_embedding), 1)

        return self.drop(self.norm(e))

class WordCNN(nn.Module):
    def __init__(self, config):
        super(WordCNN, self).__init__()

        #considering resnet18, hidden size is 224
        self.embedding = Dynamic_embeddings(config)

        in_channels = 3

        #generating conv layer following max_kernel_size param
        convModuleList = []
        for each_kernel_size in range(2, int(config.max_kernel_size)):
            convModuleList.append(nn.Conv2d(in_channels=in_channels, out_channels=config.n_channel_per_window, kernel_size=(each_kernel_size, config.hidden_size)))
        self.conv = nn.ModuleList(convModuleList)

        n_total_channels = len(self.conv) * config.n_channel_per_window

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(n_total_channels, config.label_size)

    def forward(self, x):
        # [batch_size, 3, max_seq_len, hidden_size]
        x = self.embedding(x)

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

class Word_Resnet(nn.Module):
    def __init__(self, config):
        super(Word_Resnet, self).__init__()
        
        self.embedding = Dynamic_embeddings(config)
        
        self.resnet_model = config.resnet_model
        self.fc = nn.Linear(self.resnet_model.fc.out_features, config.label_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.resnet_model(x)
        x = self.fc(x)

        return x



        

