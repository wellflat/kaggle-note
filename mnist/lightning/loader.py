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
    dataset: Dict[Phase, DataLoader]

    def setup(self, stage: Optional[str] = None) -> None:
        self.config = DataConfig()
        train = pd.read_csv(self.config.train_filepath) 
        test = pd.read_csv(self.config.test_filepath)
        train, val = self.__split_dataframe(train)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = {
            'train': MNISTDataset(train, transform),
            'val': MNISTDataset(val, transform),
            'test': MNISTDataset(test, transform) 
        }
        print(f"train: {len(self.dataset['train'])}, val: {len(self.dataset['val'])}")
    
    def train_dataloader(self) -> DataLoader:
        return self.__get_loader('train')

    def val_dataloader(self) -> DataLoader:
        return self.__get_loader('val')

    def test_dataloader(self) -> DataLoader:
        return self.__get_loader('test')
    
    def predict_dataloader(self) -> DataLoader:
        return self.__get_loader('test')

    def __get_loader(self, phase: Phase) -> DataLoader:
        loader = DataLoader(
            self.dataset[phase],
            batch_size=self.config.batch_size,
            shuffle=True if phase == 'train' else False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True if phase == "train" else False
        )
        return loader        

    def __split_dataframe(self, df:DataFrame, fraction=0.9, state=1) -> Tuple[DataFrame, DataFrame]:
        df1 = df.sample(frac=fraction, random_state=state)
        df2 = df.drop(df1.index)
        return (df1, df2)

    