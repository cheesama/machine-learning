import torch
import configparser

class LearningContainer:
    def __init__(self, config, model, criterion, dataLoader):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.dataLoader = dataLoader

