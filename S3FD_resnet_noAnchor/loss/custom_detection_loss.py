import torch
import torch.nn.functional as F

def getBBoxRegressionLoss(output, target):
    return F.smooth_l1_loss(output, target)

def getBBoxClassificationLoss(output, target):
    return F.cross_entropy(ouput, target)

def getCustomDetectionLoss(output, target):
    return getBBoxRegressionLoss(output[:4], target) + getBBoxClassificationLoss(output[4:], target)



