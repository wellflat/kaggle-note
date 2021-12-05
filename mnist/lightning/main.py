#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
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
    pl.seed_everything(36, workers=True)
    config = TrainingConfig(args.lr)
    model = Classifier(config)
    data_module = MNISTDataModule()
    logger = TensorBoardLogger('tb_logs', name='MNIST - LeNet')
    trainer_callbacks = [
        ModelCheckpoint(
            dirpath='./checkpoints',
            filename='mnist-lenet:{epoch:02d}-{val_acc:.3f}',
            monitor='val_acc',
            mode='max',
            save_top_k=1
        ),
        LearningRateMonitor('epoch')
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=trainer_callbacks,
        logger=logger
    )
    trainer.fit(model, data_module)
    print(f'best model: {trainer_callbacks[0].best_model_path}')
    trainer.save_checkpoint('mnist-lenet.ckpt')
    trainer.validate(model, data_module)