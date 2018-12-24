from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F

class SOP_resnet18_embedding(nn.Module):
    def __init__(self, embeddingnet, feature_dim=1024):
        super(SOP_resnet18_embedding, self).__init__()
        self.embeddingnet = embeddingnet
        self.feature = nn.Linear(embeddingnet.fc.out_features, feature_dim)

        nn.init.kaiming_uniform_(self.feature.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, image):
        x = self.embeddingnet(image)
        x = self.feature(x)

        feature_norm = F.normalize(x, p=2, dim=1)

        return feature_norm


if __name__ == "__main__":
    resnet18_pretrained = models.resnet18(pretrained=True)
    model = SOP_resnet18_embedding(resnet18_pretrained)

    testInput  = torch.Tensor(1,3,224,224)
    testOuput = model(testInput)


