#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=16 #this only affects MPI job
#SBATCH --time=18:00:00

module load anaconda3
source activate python_38
python3 main_downstream.py