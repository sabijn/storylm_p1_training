#!/bin/bash
#SBATCH --job-name=storylm-base
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

export WORKING_DIR="$HOME/storylm_p1_training/pipeline"
export CODE_TEMP_DIR="/scratch-shared/$USER/storylm_p1_training"

# Create temp running dir
mkdir -p "$CODE_TEMP_DIR"
rsync -a --exclude='.git' "$WORKING_DIR/" "$CODE_TEMP_DIR/"

# Create output dir
export RUN_NAME="storylm-base"
export OUTPUT_DIR="$CODE_TEMP_DIR/checkpoints/$RUN_NAME/"
mkdir -p "$OUTPUT_DIR"

# Switch working dir
cd "$CODE_TEMP_DIR"

module purge
module load 2023
module load Miniconda3/23.5.2-0
source activate babylm2026

# 1. Split the base-pretraining data into train/dev/test and cache it to disk
torchrun --nproc_per_node=1 scripts/01_prepare_data.py --config configs/data_base.yaml

# 2. Train a SentencePiece tokenizer on the train split, wrapped as a HF fast tokenizer
torchrun --nproc_per_node=1 scripts/02_train_tokenizer.py --config configs/tokenizer.yaml

# 3. Train a base model from scratch (pick gpt2 or llama sizing)
torchrun --nproc_per_node=1 scripts/03_train_base_model.py --config configs/model_base_gpt2.yaml

# 4. Evaluate the base model: dev/test loss + perplexity, and BLIMP-NL
torchrun --nproc_per_node=1 scripts/05_evaluate_model.py --config configs/eval_base.yaml


