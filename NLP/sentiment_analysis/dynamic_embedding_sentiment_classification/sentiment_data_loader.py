from torchtext.vocab import Vocab
from torchtext import data

from utils import twitter_tokenizer

import pandas as pd

def postprocess(x, train=True):
    x = int(x)
    return x

def create_data_loader(filePath, batchSize=128, tokenizer=twitter_tokenizer, build_vocab=False, device=-1):
    text_field = data.Field(tokenize=tokenizer, sequential=True, init_token='[CLS]', eos_token='[SEP]')
    label_field = data.Field(sequential=False, use_vocab=False, postprocessing=data.Pipeline(postprocess))

	dataset = data.TabularDataset(path=filePath, format='tsv', fields=[('id', None), ('document', text_field), ('label', label_field)], filter_pred=filter_pred)

	dataLoader = data.Iterator(dataset=dataset, batch_size=batchSize, sort_key=lambda x: len(x.text), train=True, repeat=False, device=device)
    
    if build_vocab:
        print('Building Vocabulary of file - ' + filePath)
        text_field.build_vocab(dataset)
        vocab_size = len(text_field.vocab)

        return dataLoader, vocab_size

    return dataLoader


