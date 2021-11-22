#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import pandas as pd
import torch
from torchsummary import summary
from loader import create_loaders
from classifier import Classifier


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
    
    conf = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'base_lr': args.lr,
        'momentum': 0.9,
        'num_classes': 10
    }
    print(conf)
    NET_PATH = './mnist_net.pth'
    loaders = create_loaders(conf)
    print(len(loaders['train']), args.batch_size)
    is_train = args.train
    estimator = Classifier(conf)
    print(summary(estimator.net, (1,28,28)))
    #sys.exit(1)
    if is_train:
        estimator.fit(loaders, conf['epochs'], resume=args.resume)
        estimator.save(NET_PATH)
    else:
        estimator.load(NET_PATH)
        result = estimator.test_nolabel(loaders['test'])
        submission = pd.read_csv('submission/sample_submission.csv')
        submission['Label'] = result
        submission.to_csv('submission.csv', index=False)        
        
