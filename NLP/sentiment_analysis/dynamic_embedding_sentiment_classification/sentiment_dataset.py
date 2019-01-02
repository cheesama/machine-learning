from utils import twitter_tokenizer

from torch.utils.data import Dataset

import pandas as pd

data = pd.read_csv('../../data/KorQuAD/KorQuAD_v1.0_train_preprocess.txt', sep='\t', header=(0))

class SentimentDataset(Dataset):
    def __init__(self, targetFolder, tokenizer=None):
        self.targetFolder = targetFolder
        self.tokenizer = tokenizer

        self.sentences = []
        self.labels = []

        for root, dirs, files in os.walk(targetFolder):
            for eachFile in files:
                data = pd.read_csv(root + os.sep + eachFile, sep='\t', header=(0))
                self.sentences.extend(data['document'].values)
                self.labels.extend(data['label'].values)

    def __getitem__(self, idx):
        if self.tokenizer is not None:
            return self.tokenizer(self.sentences[idx]), self.labels[idx]

        return self.sentences[idx], self.labels[idx]

    def __len__(self):
        return len(self.sentences)




