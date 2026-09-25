#!/bin/bash
#SBATCH --job-name=sonar-eval-100M
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

export WORKING_DIR="$HOME/storylm_p1_training/pipeline"
export CODE_TEMP_DIR="/scratch-shared/$USER/storylm_p1_training/$SLURM_JOB_ID"

# Create temp running dir
mkdir -p "$CODE_TEMP_DIR"
rsync -a --exclude='.git' "$WORKING_DIR/" "$CODE_TEMP_DIR/"

# Switch working dir
cd "$CODE_TEMP_DIR"

module purge
module load 2023
module load Miniconda3/23.5.2-0
source activate babylm2026

# 1. Evaluate trained model
torchrun --nproc_per_node=1 scripts/05_evaluate_model.py --config configs/eval_continued.yaml


