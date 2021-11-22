from typing import Dict
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
from torchvision import transforms


def create_loaders(conf: Dict) -> Dict[str, DataLoader]:
    batch_size: int = conf['batch_size']
    print('==> loading mnist dataset')
    transform = transforms.Compose([
        #transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train = pd.read_csv('train.csv') 
    test = pd.read_csv('test.csv')
    print('train data:' + str(train.shape))
    print('test data:' + str(test.shape))
    train_ds = train.drop('label', axis=1).values
    test_ds = test.values
    labels = train.label.to_numpy()
    train_tensor = torch.tensor(train_ds)
    test_tensor = torch.tensor(test_ds)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    train_dataset = TensorDataset(train_tensor, labels_tensor)
    split = (40000, 2000) # (32000, 10000)
    train_set , val_set = random_split(train_dataset, split)
    test_set = TensorDataset(test_tensor)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def create_loaders_from_torchvision(conf: Dict) -> Dict[str, DataLoader]:
    batch_size: int = conf['batch_size']
    print('==> loading mnist dataset')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
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