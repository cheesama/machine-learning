import torch, torchvision
from torchvision.transforms import transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class Cifar10ImageLoader:
    def __init__(self, data_dir='./data', batch_size=128, train_shuffle=True, val_shuffle=False, num_workers=4):
        self.trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        self.trainLoader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)

        self.testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        self.testLoader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=val_shuffle, num_workers=num_workers)
