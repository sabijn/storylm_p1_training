#!/bin/bash
#SBATCH --job-name=sonar-extract
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

export WORKING_DIR="$HOME/storylm_p1_training/pipeline"
# Job-specific checkout dir (not shared across concurrent jobs, unlike run.sh's fixed path).
export CODE_TEMP_DIR="/scratch-shared/$USER/storylm_p1_training/$SLURM_JOB_ID"

mkdir -p "$CODE_TEMP_DIR"
rsync -a --exclude='.git' "$WORKING_DIR/" "$CODE_TEMP_DIR/"
cd "$CODE_TEMP_DIR"

module purge
module load 2023
module load Miniconda3/23.5.2-0
source activate babylm2026

# Run once. After this, category selection / sampling happens via the `local` source
# type in data_continued.yaml - no need to rerun this script for a different subset.
python scripts/prepare_sonar.py --config configs/sonar_extract.yaml
