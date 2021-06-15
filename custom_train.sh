#!/bin/bash

echo "1"
mkdir ./checkpoints/roya_1
python3 -u train.py --name roya-1 --stage syntheticCLR --substage 1 --validation syntheticCLR --restore_ckpt models/raft-chairs.pth --gpus 0 --num_steps 10000 --batch_size 2 --lr 0.00005 --image_size 320 720 --wdecay 0.00001 --gamma=0.85 --mixed_precision
mv ./checkpoints/*.pth ./checkpoints/roya_1

echo "2"
mkdir ./checkpoints/roya_2
python3 -u train.py --name roya-2 --stage syntheticCLR --substage 2 --validation syntheticCLR --restore_ckpt checkpoints/roya_1/roya-1.pth --gpus 0 --num_steps 10000 --batch_size 2 --lr 0.00005 --image_size 320 720 --wdecay 0.00001 --gamma=0.85 --mixed_precision
mv ./checkpoints/*.pth ./checkpoints/roya_2

echo "3"
mkdir ./checkpoints/roya_3
python3 -u train.py --name roya-3 --stage syntheticCLR --substage 3 --validation syntheticCLR --restore_ckpt checkpoints/roya_2/roya-2.pth --gpus 0 --num_steps 10000 --batch_size 2 --lr 0.00005 --image_size 320 720 --wdecay 0.00001 --gamma=0.85 --mixed_precision
mv ./checkpoints/*.pth ./checkpoints/roya_3

#echo "4"
#mkdir ./checkpoints/roya_4
#python3 -u train.py --name roya-4 --stage syntheticCLR --substage 4 --validation syntheticCLR --restore_ckpt checkpoints/roya_3/roya-3.pth --gpus 0 --num_steps 10000 --batch_size 4 --lr 0.00005 --image_size 320 720 --wdecay 0.00001 --gamma=0.85 --mixed_precision
#mv ./checkpoints/*.pth ./checkpoints/roya_4

echo "5"
mkdir ./checkpoints/roya_5
python3 -u train.py --name roya-5 --stage syntheticCLR --substage 5 --validation syntheticCLR --restore_ckpt checkpoints/roya_3/roya-3.pth --gpus 0 --num_steps 10000 --batch_size 2 --lr 0.00005 --image_size 320 720 --wdecay 0.00001 --gamma=0.85 --mixed_precision
mv ./checkpoints/*.pth ./checkpoints/roya_5
