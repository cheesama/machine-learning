import torch
import torch.nn as nn

from torchvision import models

from .detection_module import YOLOLayer

from collections import defaultdict

class YoloV3_resnet(nn.Module):
    def __init__(self, embeddingnet, input_size=416):
        super(YoloV3_resnet, self).__init__()

        self.embeddingnet = embeddingnet
        self.losses = defaultdict(float)
        self.loss_names = ['x', 'y', 'w', 'h', 'conf', 'cls', 'recall']
    
        anchor = [(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),
        (156,198),  (373,326)]

        #modifying from paper anchor set(based on resnet152)
        self.yolo1_layer = YOLOLayer([anchor[i] for i in range(5,9)])
        self.yolo1_channel_reduction = nn.Conv2d(512, self.yolo1_layer.num_anchors * self.yolo1_layer.bbox_attrs, kernel_size=1)

        self.yolo2_layer = YOLOLayer([anchor[i] for i in range(3,7)])
        self.yolo2_channel_reduction = nn.Conv2d(1024, self.yolo2_layer.num_anchors * self.yolo2_layer.bbox_attrs, kernel_size=1)

        self.yolo3_layer = YOLOLayer([anchor[i] for i in range(0,4)])
        self.yolo3_channel_reduction = nn.Conv2d(2048, self.yolo3_layer.num_anchors * self.yolo3_layer.bbox_attrs, kernel_size=1)

    def forward(self, image, target=None):
        output = []
        if target is not None:
            is_training = True
        else:
            is_training = False

        x = self.embeddingnet.conv1(image)
        x = self.embeddingnet.bn1(x)
        x = self.embeddingnet.relu(x)
        x = self.embeddingnet.maxpool(x)

        x = self.embeddingnet.layer1(x)
        x = self.embeddingnet.layer2(x) 
        if is_training: 
            output1_pred = self.yolo1_channel_reduction(x)
            output1_loss, *losses = self.yolo1_layer(output1_pred, target)
            output.append(output1_loss)
            for name, loss in zip(self.loss_names, losses):
                self.losses[name] += loss

            x = self.embeddingnet.layer3(x)
            output2_pred = self.yolo2_channel_reduction(x)
            output2_loss, *losses = self.yolo2_layer(output2_pred, target)
            output.append(output2_loss)
            for name, loss in zip(self.loss_names, losses):
                self.losses[name] += loss

            x = self.embeddingnet.layer4(x)
            output3_pred = self.yolo3_channel_reduction(x)
            output3_loss, *losses = self.yolo3_layer(output3_pred, target)
            output.append(output3_loss)
            for name, loss in zip(self.loss_names, losses):
                self.losses[name] += loss
        else:
            output1_pred = self.yolo1_channel_reduction(x)
            output1_pred = self.yolo1_layer(output1_pred, target)
            output.append(output1_pred)

            x = self.embeddingnet.layer3(x)
            output2_pred = self.yolo2_channel_reduction(x)
            output2_pred = self.yolo2_layer(output2_pred, target)
            output.append(output2_pred)

            x = self.embeddingnet.layer4(x)
            output3_pred = self.yolo3_channel_reduction(x)
            output3_pred = self.yolo3_layer(output3_pred, target)
            output.append(output3_pred)

        self.losses['recall'] /= 3

        #To-do: return all loss values seperately
        return sum(output) if is_training else torch.cat(output, 1)









        














