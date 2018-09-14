import torch
import torch.nn as nn
from torchvision import models

class SFD_resnet(nn.Module):
    def __init__(self, embeddingnet):
        super(SFD_resnet, self).__init__()
        self.embeddingnet = embeddingnet

        self.output1_conv_block1 = self._conv_block(64,5)
        self.output1_conv_block2 = self._conv_block(5,5)
        self.output1_conv_block3 = self._conv_block(5,5)
        self.output1_conv_block4 = self._conv_block(5,5)
        self.output1_conv_block5 = self._conv_block(5,5)

        self.output2_conv_block1 = self._conv_block(128,5)
        self.output2_conv_block2 = self._conv_block(5,5)
        self.output2_conv_block3 = self._conv_block(5,5)
        self.output2_conv_block4 = self._conv_block(5,5)

        self.output3_conv_block1 = self._conv_block(256,5)
        self.output3_conv_block2 = self._conv_block(5,5)
        self.output3_conv_block3 = self._conv_block(5,5)

        self.output4_conv_block1 = self._conv_block(512,5)
        self.output4_conv_block2 = self._conv_block(5,5)

    def _conv_block(self, in_channel, out_channel, kernel=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel, padding=kernel // 2,
            stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=kernel, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.embeddingnet.conv1(x)
        x = self.embeddingnet.bn1(x)
        x = self.embeddingnet.relu(x)
        x = self.embeddingnet.maxpool(x)

        x = self.embeddingnet.layer1(x)
        output1 = self.output1_conv_block1(x)
        output1 = self.output1_conv_block2(output1)
        output1 = self.output1_conv_block3(output1)
        output1 = self.output1_conv_block4(output1)
        output1 = self.output1_conv_block5(output1)

        x = self.embeddingnet.layer2(x)
        output2 = self.output2_conv_block1(x)
        output2 = self.output2_conv_block2(output2)
        output2 = self.output2_conv_block3(output2)
        output2 = self.output2_conv_block4(output2)

        x = self.embeddingnet.layer3(x)
        output3 = self.output3_conv_block1(x)
        output3 = self.output3_conv_block2(output3)
        output3 = self.output3_conv_block3(output3)

        x = self.embeddingnet.layer4(x)
        output4 = self.output4_conv_block1(x)
        output4 = self.output4_conv_block2(output4)

        return [output1, output2, output3, output4]
