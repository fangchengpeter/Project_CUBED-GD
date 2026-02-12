#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=com
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=70:00:00
#SBATCH --output=slurm.%N.%j.log
#SBATCH --error=slurm.%N.%j.log


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /home/jc3585/anaconda3/etc/profile.d/conda.sh
conda activate combridge
cd /scratch/jc3585/com
echo "Running with conda env: ($CONDA_DEFAULT_ENV)"
echo "Python path: $(which python)"

srun python3 main.py
