import torch
import torch.nn as nn
import configparser
import mlflow

from container import LearningContainer

#set custom model * data_loader
from model import cifar10_classification_model
from loader import cifar10_image_loader

config = configparser.ConfigParser()
config.read('config.ini')
config = config['cifar10'] #section config load

model = cifar10_classification_model.Cifar10_classifier()
if config['multiGPU']=='Y':
    model = nn.DataParallel(model).cuda()
else:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

dataLoader = cifar10_image_loader.Cifar10ImageLoader(data_dir=config['data_dir'], batch_size=int(config['batch_size']))

learningContainer = LearningContainer(config, model, criterion, dataLoader)
