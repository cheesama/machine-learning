###set custom model & data_loader & criterion & metric classes
from model import cifar10_classification_model
from loader import cifar10_image_loader
from metric.Metric import Accuracy, MSE

import torch.nn as nn #re-use pre defined pytorch criterion

#set the model
model = cifar10_classification_model.Cifar10_classifier()

#set the dataLoader
dataLoader = cifar10_image_loader.Cifar10ImageLoader(data_dir='../data/image', batch_size=128)

#set the loss function(if you implement your own, import that custom loss class)
criterion = nn.CrossEntropyLoss()

customMetric = Accuracy

from autorch import train_eval

tuning = train_eval.Tuning()

#set the learning config
tuning.setConfigFile('cifar10','config.ini')

tuning.setCustomModel(model)
tuning.setCustomDataLoader(dataLoader)
tuning.setCriterion(criterion)
tuning.setCustomMetric(customMetric)

tuning.fit()



