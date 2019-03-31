from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

import sys
import collections
import math
import time
import random

with open('data/ptb.train.txt','r') as f:
    lines = f.readlines()
    raw_dataset = [st.split() for st in lines]

print ('# sentences: {}'.format(len(raw_dataset)))

# create word index
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x:x[1] >= 5, counter.items()))

idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk : idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx] for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])
print ('# tokens: {}'.format(num_tokens))

# subsampling
def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / counter[idx_to_token[idx]] * num_tokens)

subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
print ('# subsampled tokens: {}'.format(sum([len(st) for st in subsampled_dataset])))

def compare_counts(token):
    before_count = sum([st.count(token_to_idx[token]) for st in dataset])
    after_count = sum([st.count(token_to_idx[token]) for st in subsampled_dataset])

    print ('token: {}, before count: {}, after_count: {}'.format(token, before_count, after_count))

#discard tokens test
compare_counts('join')
compare_counts('the')

def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []

    for st in dataset:
        if len(st) < 2: continue

        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size), min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)
            contexts.append([st[idx] for idx in indices])

    return centers, contexts

# getting center & context test
tiny_dataset = [list(range(7)), list(range(7,10))]
print ('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print ('center', center, 'has contexts', context)

# negative sampling
def get_negatives(all_contexts, sampling_weights, K):



            
            

