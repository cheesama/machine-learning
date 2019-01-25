from torchtext.vocab import Vocab
from torchtext import data

from utils import twitter_tokenizer

import pandas as pd

def postprocess(x, train=True):
    x = int(x)
    return x

def create_data_loader(train_file_path, val_file_path, batchSize=128, tokenizer=twitter_tokenizer, device=-1):
    text_field = data.Field(tokenize=tokenizer, sequential=True, init_token='[CLS]', eos_token='[SEP]')
    label_field = data.Field(sequential=False, use_vocab=False, postprocessing=data.Pipeline(postprocess))

    train_dataset = data.TabularDataset(train_file_path, format='tsv', fields=[('id', None), ('document', text_field), ('label', label_field)], filter_pred=lambda ex: ex.label in ['0', '1'])
        
    print('Building Vocabulary of file - ' + train_file_path)
    text_field.build_vocab(train_dataset)

    vocab_size = len(text_field.vocab)

    train_dataLoader = data.Iterator(dataset=train_dataset, batch_size=batchSize, sort_key=lambda x: len(x.document), train=True, repeat=False, device=device)

    val_dataset = data.TabularDataset(val_file_path, format='tsv', fields=[('id', None), ('document', text_field), ('label', label_field)], filter_pred=lambda ex: ex.label in ['0', '1'])
    val_dataLoader = data.Iterator(dataset=val_dataset, batch_size=batchSize, sort=False, train=False, device=device)
    
    return train_dataLoader, val_dataLoader, vocab_size


