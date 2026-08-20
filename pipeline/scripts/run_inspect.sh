#!/bin/bash
#SBATCH --job-name=storylm-base
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=00:05:00
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

python scripts/06_inspect_model.py --config configs/eval_base.yaml



