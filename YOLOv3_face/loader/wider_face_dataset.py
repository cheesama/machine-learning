import os, sys
import numpy as np

import torch

from torch.utils.data import Dataset

from tqdm import tqdm
from PIL import Image
from skimage.transform import resize

class WiderFaceDataset(Dataset):
    def __init__(self, image_dir_path, annotation_path, img_size=416, transform=None):
        super().__init__()
        self.image_dir_path = image_dir_path
        self.transform = transform
        self.img_shape = (img_size, img_size)

        self.imagePathList = []
        self.bboxList = []

        print ('Reading annotation file & mapping images')
        with open(annotation_path, 'r') as annotationFile:
            while True:
                line = annotationFile.readline()
                if not line: break

                imagePath = line.strip()
                bboxNum = int(annotationFile.readline().strip())
                if bboxNum == 0:
                    continue

                bboxInfo = np.zeros((bboxNum, 5))
                for i in range(bboxNum):
                    eachbboxInfo = annotationFile.readline()
                    x1 = float(eachbboxInfo.split()[0])
                    y1 = float(eachbboxInfo.split()[1])
                    x2 = x1 + float(eachbboxInfo.split()[2])
                    y2 = y1 + float(eachbboxInfo.split()[3])
                    cls = int(eachbboxInfo.split()[7])
                    bboxInfo[i] = [cls, x1, y1, x2, y2]
                    self.imagePathList.append(image_dir_path + os.sep + imagePath)
                    self.bboxList.append(bboxInfo)

        print ('image & annotatation mapping done: ' + str(len(self.bboxList)) + ' annotations exist.')
                    
    def __len__(self):
        return len(self.bboxList)

    def __getitem__(self, index):
        img = np.array(Image.open(self.imagePathList[index]))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        labels = self.bboxList[index]
        x1 = labels[:, 1]
        y1 = labels[:, 2]
        x2 = labels[:, 3]
        y2 = labels[:, 4]

        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates
        labels[:, 1] = ((x1 + x2) / 2) / padded_w
        labels[:, 2] = ((y1 + y2) / 2) / padded_h
        labels[:, 3] *= w / padded_w
        labels[:, 4] *= h / padded_h

        labels = torch.from_numpy(labels)

        if self.transform:
            #following transform, bbox also should be transformed(if scale or transition)
            input_img = self.transform(input_img)

        return self.imagePathList[index], input_img, labels

