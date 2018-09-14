import torch, torchvision
from torchvision.transforms import transforms

from . import wider_face_dataset

#set the training transform(for augmentation, it depends on pretrained model)
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class WiderFaceDataLoader:
    def __init__(self, train_image_dir, test_image_dir, train_annotation_path, test_annotation_path, batch_size=128, train_transform=data_transforms['train'], test_transform=data_transforms['val'], num_workers=4):
        #set the dataLoader
        self.trainset = wider_face_dataset.WiderFaceDataset(train_image_dir, train_annotation_path,
        transform=train_transform)
        self.trainLoader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)

        self.testset = wider_face_dataset.WiderFaceDataset(test_image_dir, test_annotation_path,
        transform=test_transform)
        self.testLoader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
