import os, sys
import random
import numpy as np

from torch.utils.data import Dataset

from PIL import Image
from tqdm import tqdm

class SOPDataset(Dataset):
    def __init__(self, targetFolder, targetFile, transform=None):
        self.targetFolder = targetFolder
        self.targetFile = targetFile
        self.transform = transform

        #load image path from file -> key: class, value(list): file list
        self.image_list = {}
        self.shuffled_class_list = []
        self.shuffled_image_list = []

        with open(self.targetFolder + os.sep + self.targetFile, 'r') as inputFile:
            line = inputFile.readline()
            while True:
                line = inputFile.readline()
                if not line: break
                
                class_id = line.strip().split()[1]
                eachFilePath = line.strip().split()[3]
                if class_id not in self.image_list.keys():
                    self.image_list[int(class_id)] = []

                self.image_list[int(class_id)].append(self.targetFolder + os.sep + line.strip().split()[3])

        self.shuffle()

    def shuffle(self):
        class_idx = list(range(min(self.image_list.keys()), max(self.image_list.keys())+1))
        random.shuffle(class_idx)

        print ('class shuffling')
        for eachClass in tqdm(class_idx):
            for eachImagePath in self.image_list[eachClass]:
                self.shuffled_class_list.append(eachClass)
                self.shuffled_image_list.append(eachImagePath)

    def __getitem__(self, index):
        image = Image.open(self.shuffled_image_list[index]).convert(mode='RGB')
        class_id = self.shuffled_class_list[index]

        if self.transform:
            image = self.transform(image)

        return image, class_id

    def __len__(self):
        return len(self.shuffled_image_list)


if __name__ == "__main__":
    SOPTrainDataset = SOPDataset('data/Stanford_Online_Products','Ebay_train.txt') 
    SOPValDataset = SOPDataset('data/Stanford_Online_Products','Ebay_test.txt') 

