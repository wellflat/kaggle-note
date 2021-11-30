from typing import Dict, Tuple, Literal, Optional
import pandas as pd
from pandas import DataFrame
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from config import DataConfig
from dataset import MNISTDataset

Phase = Literal['train', 'val', 'test']

class MNISTDataModule(pl.LightningDataModule):
    config: DataConfig
    transform: transforms.Compose
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.config = DataConfig()
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = self.__prepare_data()
    
    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.dataset['train'],
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.dataset['val'],
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.dataset['test'],
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        return loader

    def __get_loader(self, phase: str) -> DataLoader:
        loader = DataLoader(
            self.dataset[phase],
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        return loader
    
    def __prepare_data(self) -> Dict[str, DataFrame]:
        train = pd.read_csv(self.config.train_filepath) 
        test = pd.read_csv(self.config.test_filepath)
        train, val = self.__split_dataframe(train)
        return {
            'train': train,
            'val': val,
            'test': test
        }

    def __split_dataframe(self, df:DataFrame, fraction=0.9, state=1) -> Tuple[DataFrame, DataFrame]:
        df1 = df.sample(frac=fraction, random_state=state)
        df2 = df.drop(df1.index)
        return (df1, df2)

    