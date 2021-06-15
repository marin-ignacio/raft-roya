from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets_test

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def train(args):

    train_loader = datasets_test.fetch_dataloader(args)

    total_steps = 0

    should_keep_training = True

    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):

            print("Epoch {}".format(i_batch))
    
            image1, image2, flow, valid, image1_name, image2_name = [x.cuda() if (i < 4) else x for i, x in enumerate(data_blob)]

            print("image1 = {}".format(image1_name))
            print("image2 = {}".format(image2_name))
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

            print()

    PATH = 'checkpoints/%s.pth' % args.name

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")

    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])

    args = parser.parse_args()

    #torch.manual_seed(1234)
    #np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)
                                   