import os, sys
import pandas as pd
import numpy as np

from konlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

targetFolder = '../data/voc_stt_sample'
outputFileName = 'voc_preprocessed.txt'
sentence_threshold = 10
token_threshold = 3

dialogs = []

tokenizer = Twitter()

def twitter_tokenizer(sentence):
    word_array = []

    #infer http://konlpy.org/en/v0.4.4/api/konlpy.tag/
    for eachPart in tokenizer.pos(sentence, norm=True, stem=True):
        word = eachPart[0]
        part = eachPart[1]

        if part == 'Alpha' or part == 'Punctuation':
            continue

        word_array.append(word + '/' + part)
        
    return word_array

print ('preprocessing')
for root, dirs, files in os.walk(targetFolder):
    for eachFile in tqdm(files):
        eachDialog = []

        turns = open(root + os.sep + eachFile).readlines()

        #skip short conversation talk
        if len(turns) < sentence_threshold // 2:
            continue

        for eachTurn in turns:
            a_sentence = twitter_tokenizer(eachTurn.split('\t')[1].strip())
            b_sentence = twitter_tokenizer(eachTurn.split('\t')[3].strip())

            if len(a_sentence) >= token_threshold:
                eachDialog += a_sentence
        
            if len(b_sentence) >= token_threshold:
                eachDialog += b_sentence

        dialogs.append(eachDialog)

print ('saving')
with open(outputFileName, 'w') as outputFile:
    for eachDialog in dialogs:
        outputFile.write(' '.join(eachDialog))
        outputFile.write('\n')



