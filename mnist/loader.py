from typing import Dict, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
from torchvision import transforms
from dataset import MNISTDataset


def split_dataframe(df:pd.DataFrame, fraction=0.95, state=1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df1 = df.sample(frac=fraction, random_state=state)
    df2 = df.drop(df1.index)
    return (df1, df2)

def create_loaders(conf: Dict) -> Dict[str, DataLoader]:
    batch_size: int = conf['batch_size']
    print('==> loading mnist dataset')
    train = pd.read_csv('train.csv') 
    test = pd.read_csv('test.csv')
    print('train data:' + str(train.shape))
    print('test data:' + str(test.shape))
    train, val = split_dataframe(train)
    transform = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    }
    train_set = MNISTDataset(train, transform['train'])
    val_set = MNISTDataset(val, transform['val'])
    test_set = MNISTDataset(test, transform['val'])
    print(len(train_set),len(val_set),len(test_set))
    #split = (40000, 2000) # (32000, 10000)
    #gen = torch.Generator().manual_seed(784)
    #train_set, val_set = random_split(dataset, split, generator=gen)
    
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