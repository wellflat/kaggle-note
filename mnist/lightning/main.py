#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from loader import MNISTDataModule
from classifier import Classifier
from config import TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Classifier Training')
    parser.add_argument('--train', '-t', action='store_true', default=False, help='training mode')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    config = TrainingConfig(args.lr)
    model = Classifier(config)
    data_module = MNISTDataModule()
    logger = TensorBoardLogger('tb_logs', name='MNIST - LeNet')
    root_dir = './checkpoints'
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        max_epochs=args.epochs,
        logger=logger
    )
    trainer.fit(model, data_module)
    #trainer.validate(model, data_module)