#!/bin/bash
#SBATCH --job-name=storylm-base-32k-run3
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:30:00
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

# 1. Split the base-pretraining data into train/dev/test and cache it to disk
torchrun --nproc_per_node=1 scripts/01_prepare_data.py --config configs/data_base.yaml

# 2. Train a base model from scratch (pick gpt2 or llama sizing)
torchrun --nproc_per_node=1 scripts/03_train_base_model.py --config configs/model_base_gpt2.yaml

# 3. Evaluate the base model: dev/test loss + perplexity, and BLIMP-NL
torchrun --nproc_per_node=1 scripts/05_evaluate_model.py --config configs/eval_base.yaml

# 4. Prepare a different dataset, then continue pretraining the base model on it
torchrun --nproc_per_node=1 scripts/01_prepare_data.py --config configs/data_continued.yaml
torchrun --nproc_per_node=1 scripts/04_continue_pretraining.py --config configs/model_continued.yaml

# 5. Evaluate the continued-pretraining model the same way
torchrun --nproc_per_node=1 scripts/05_evaluate_model.py --config configs/eval_continued.yaml


