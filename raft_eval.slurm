#!/bin/bash
#SBATCH --job-name=raft_eval
#SBATCH --output=result.txt
#SBATCH --partition=nukwa-long
#SBATCH --nodelist=nukwa-04.cnca
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --mail-user=maxhr08@estudiantec.cr
#SBATCH --mail-type=BEGIN,END,FAIL

srun python3 evaluate.py --model=models/raft-chairs.pth --dataset=sintel --mixed_precision
