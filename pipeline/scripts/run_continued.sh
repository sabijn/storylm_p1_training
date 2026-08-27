#!/bin/bash
#SBATCH --job-name=storylm-continue-16k-run3
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

export WORKING_DIR="$HOME/storylm_p1_training/pipeline"
export CODE_TEMP_DIR="/scratch-shared/$USER/storylm_p1_training"

# Create temp running dir
mkdir -p "$CODE_TEMP_DIR"
rsync -a --exclude='.git' "$WORKING_DIR/" "$CODE_TEMP_DIR/"

# Switch working dir
cd "$CODE_TEMP_DIR"

module purge
module load 2023
module load Miniconda3/23.5.2-0
source activate babylm2026

# 5. Prepare a different dataset, then continue pretraining the base model on it
python scripts/01_prepare_data.py --config configs/data_continued.yaml
python scripts/04_continue_pretraining.py --config configs/model_continued.yaml

# 6. Evaluate the continued-pretraining model the same way
python scripts/05_evaluate_model.py --config configs/eval_continued.yaml


