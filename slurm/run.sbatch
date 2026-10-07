#!/bin/bash
#SBATCH --job-name=Pipat_RR
#SBATCH --output=logs/RR/%j.log
#SBATCH --error=logs/RR/%j.err
#SBATCH --partition=gpu4090
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

DIRECTORY_VENV_ACTIVATE="$HOME/RR/.venv/bin/activate"
DIRECTORY_LAB="$HOME/RR/src"

set -euo pipefail

cd $HOME

START_TIME=$SECONDS

print_elapsed() {
    local elapsed=$(( SECONDS - START_TIME ))
    printf "Elapsed time: %02d:%02d:%02d\n" \
        $(( elapsed / 3600 )) $(( (elapsed % 3600) / 60 )) $(( elapsed % 60 ))
}
trap print_elapsed EXIT

echo "=== Host ==="
echo "Job executed on: $(hostname)"
echo "Job started on: $(date)"

source $DIRECTORY_VENV_ACTIVATE

echo "=== Python ==="
echo "Python executable location: $(which python)"
echo "Python version: $(python --version)"

echo "=== GPU ==="
nvidia-smi

echo "=== Execution ==="
python "$DIRECTORY_LAB/rr.py"

echo "Job finished on: $(date)"