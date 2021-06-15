#!/bin/bash
#SBATCH --job-name=raft
#SBATCH --output=result.txt
#SBATCH --partition=nukwa
#SBATCH --nodelist=nukwa-04.cnca
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mail-user=maxhr08@estudiantec.cr
#SBATCH --mail-type=BEGIN,END,FAIL


python3 -u train.py --name raft-sintel --stage sintel --validation sintel --restore_ckpt models/raft-things.pth --gpus 0 --num_steps 500000 --batch_size 2 --lr 0.00005 --image_size 640 360 --wdecay 0.00001 --gamma=0.8 --mixed_precision --dropout 0.1
