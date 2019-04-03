from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tqdm import tqdm

import mxnet as mx
import sys
import collections
import math
import time
import random

with open('data/tworld_QA_dataset.csv','r') as f:
    lines = f.readlines()
    raw_dataset = [st.split() for st in lines[1:]]

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
compare_counts('네')
compare_counts('지원금')

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

all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

# negative sampling
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in tqdm(all_contexts, desc='getting negative samples'):
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                i, neg_candidates = 0, random.choices(population, sampling_weights, k=int(1e5))
            
            neg, i = neg_candidates[i], i + 1

            if neg not in set(contexts):
                negatives.append(neg)

        all_negatives.append(negatives)

    return all_negatives

sampling_weights = [counter[w] ** 0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)

# Reading Data
def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * ( max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]

    return (nd.array(centers).reshape((-1, 1)), nd.array(contexts_negatives), nd.array(masks), nd.array(labels))


batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4
dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True, batchify_fn = batchify, num_workers=num_workers)

for batch in data_iter:
    for name, data in zip(['centers','contexts_negatives','masks','labels'], batch):
        print (name, 'shape:', data.shape)
    break

# Skip-Gram Model
## Embedding Layer test
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
print (embed.weight)

x = nd.array([[1,2,3],[4,5,6]])
print (embed(x))

## batch_dot test
X = nd.ones((2, 1, 4))
Y = nd.ones((2, 4, 6))
print (nd.batch_dot(X, Y).shape)

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = nd.batch_dot(v, u.swapaxes(1, 2))
    return pred


## loss test
loss_fn = gloss.SigmoidBinaryCrossEntropyLoss()
pred = nd.array([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
# 1 and 0 in the label variables label represent context words and the noise
# words, respectively
label = nd.array([[1, 0, 0, 0], [1, 1, 0, 0]])
mask = nd.array([[1, 1, 1, 1], [1, 1, 1, 0]])  # Mask variable
print (loss_fn(pred, label, mask) * mask.shape[1] / mask.sum(axis=1))

def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print('%.7f' % ((sigmd(1.5) + sigmd(-0.3) + sigmd(1) + sigmd(-2)) / 4))
print('%.7f' % ((sigmd(1.1) + sigmd(-0.6) + sigmd(-2.2)) / 3))

## Init Model Params
embed_size = 768
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size), nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size))

def train(net, lr, num_epochs):
    ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})

    for epoch in range(num_epochs):
        start, loss_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [data.as_in_context(ctx) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                loss = (loss_fn(pred.reshape(label.shape), label, mask) * mask.shape[1] / mask.sum(axis=1))
            loss.backward()
            trainer.step(batch_size)
            loss_sum += loss.sum().asscalar()
            n += loss.size
        print ('epoch %d, loss %.2f, tim %.2fs' % (epoch + 1, loss_sum / n, time.time() - start))


train (net, 0.005, 10)
