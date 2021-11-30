from dataclasses import dataclass

@dataclass
class DataConfig:
    train_filepath: str = './train.csv'
    test_filepath: str = './test.csv'
    batch_size: int = 100
    num_workers: int = 0
