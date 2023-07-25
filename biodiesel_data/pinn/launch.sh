#!/bin/bash
#SBATCH --account=def-blaisbru
#SBATCH --gpus-per-node=1
#SBATCH --mem=5000M               # memory per node
#SBATCH --time=12:00:00
source $SCRATCH/identification/ENV/bin/activate
srun python3 main.py
