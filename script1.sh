#!/bin/bash
#SBATCH -A seshadri_c
#SBATCH -c 10
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file.txt
#SBATCH --nodelist=gnode85

source ~/home/projects/FSLT/slt_env/bin/activate

python train.py

