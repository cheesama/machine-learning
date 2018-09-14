import os, sys
import torch
import cv2

from torch.utils.data import Dataset

from tqdm import tqdm

class WiderFaceDataset(Dataset):
    def __init__(self, image_dir_path, annotation_path, rescale_size=640, transform=None):
        super().__init__()
        self.image_dir_path = image_dir_path
        self.transform = transform
        self.rescale_size = rescale_size

        self.imagePathList = []
        self.bboxList = []

        print ('Reading annotation file & mapping images')
        with open(annotation_path, 'r') as annotationFile:
            while True:
                line = annotationFile.readline()
                if not line: break

                imagePath = line.strip()
                bboxNum = int(annotationFile.readline().strip())
                for i in range(bboxNum):
                    bboxInfo = annotationFile.readline()
        
                    x1 = float(bboxInfo.split()[0])
                    y1 = float(bboxInfo.split()[1])
                    x2 = x1 + float(bboxInfo.split()[2])
                    y2 = y1 + float(bboxInfo.split()[3])
                    cls = int(bboxInfo.split()[7])

                    self.imagePathList.append(image_dir_path + os.sep + imagePath)
                    self.bboxList.append([x1, y1, x2, y2, cls])

        print ('image & annotatation mapping done: ' + str(len(self.bboxList)) + ' annotations exist.')
                    
    def __len__(self):
        return len(self.bboxList)

    def __getitem__(self, index):
        #scale coordinate
        width = self.bboxList[index][2] - self.bboxList[index][0]
        height = self.bboxList[index][3] - self.bboxList[index][1]
        width_scale, height_scale = self.rescale_size / width, self.rescale_size / height
        coordinates = [self.bboxList[index][0] * width_scale, self.bboxList[index][1] * height_scale, self.bboxList[index][2] * width_scale, self.bboxList[index][3] * height_scale]

        image = cv2.imread(self.imagePathList[index])
        image = cv2.resize(image, (self.rescale_size, self.rescale_size))
        if self.transform:
            #following transform, bbox also should be transformed(if scale or transition)
            image = self.transform(image)

        return image, coordinates, self.imagePathList[index]

