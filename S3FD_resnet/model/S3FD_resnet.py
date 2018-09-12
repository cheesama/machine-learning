import torch
import torch.nn as nn
from torchvision import models

class SFD_resnet(nn.Module):
    def __init__(self, embeddingnet):
        super(SFD_resnet, self).__init__()
        self.embeddingnet = embeddingnet

        self.prediction1_reg = nn.Conv2d(64,4,kernel_size=3,padding=1)
        self.prediction1_cls = nn.Conv2d(64,2,kernel_size=3,padding=1)

        self.prediction2_reg = nn.Conv2d(128,4,kernel_size=3,padding=1)
        self.prediction2_cls = nn.Conv2d(128,2,kernel_size=3,padding=1)

        self.prediction3_reg = nn.Conv2d(256,4,kernel_size=3,padding=1)
        self.prediction3_cls = nn.Conv2d(256,2,kernel_size=3,padding=1)

        self.prediction4_reg = nn.Conv2d(512,4,kernel_size=3,padding=1)
        self.prediction4_cls = nn.Conv2d(512,2,kernel_size=3,padding=1)

    def forward(self, x):
        x = self.embeddingnet.conv1(x)
        x = self.embeddingnet.bn1(x)
        x = self.embeddingnet.relu(x)
        x = self.embeddingbet.maxpool(x)

        x = self.embeddingnet.layer1(x)
        output1_reg = self.prediction1_reg(x)
        output1_cls = self.prediction1_cls(x)

        x = self.embeddingnet.layer2(x)
        output2_reg = self.prediction2_reg(x)
        output2_cls = self.prediction2_cls(x)

        x = self.embeddingnet.layer1(x)
        output3_reg = self.prediction1_reg(x)
        output3_cls = self.prediction1_cls(x)

        x = self.embeddingnet.layer1(x)
        output4_reg = self.prediction4_reg(x)
        output4_cls = self.prediction4_cls(x)

        return [output1_reg, output1_cls, output2_reg, output2_cls, output3_reg, output3_cls, output4_reg, output4_cls]



