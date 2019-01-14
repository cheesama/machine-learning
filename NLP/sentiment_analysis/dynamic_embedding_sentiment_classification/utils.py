import os, sys
import pandas as pd
import numpy as np

from konlpy.tag import Twitter
from tqdm import tqdm

tokenizer = Twitter()

def twitter_tokenizer(sentence):
    word_array = []

    #following 'BERT' tokenization
    word_array.append('[CLS]')

    #infer http://konlpy.org/en/v0.4.4/api/konlpy.tag/
    for eachPart in tokenizer.pos(sentence, norm=True):
        word = eachPart[0]
        part = eachPart[1]

        if part == 'Alpha':
            continue

        word_array.append(word + '/' + part)

    #following 'BERT' tokenization
    word_array.append('[SEP]')

    return word_array


