from typing import Dict
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


def create_loaders(conf: Dict) -> Dict[str, DataLoader]:
    batch_size: int = conf['batch_size']
    print('==> loading cifar10 dataset')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    testset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testloader = DataLoader(
        testset, batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(trainset.data.shape)
    print(testset.data.shape)
    print(trainset.classes)

    return {
        'train': trainloader,
        'val': testloader
    }