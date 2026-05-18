#!/bin/bash
#SBATCH --job-name=sadhwani-mortgage
#SBATCH --output=logs/sadhwani_%j.out
#SBATCH --error=logs/sadhwani_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Sadhwani et al. (2021) — GPU Cluster Training
#
# Adjust #SBATCH directives above for your cluster:
#   --partition   : gpu partition name (e.g., gpu, gpu_a100, gpu_v100)
#   --gres        : GPU count/type (e.g., gpu:a100:1, gpu:v100:2)
#   --mem         : RAM (64G should suffice for ~5M rows)
#   --time        : wall time (ensemble of 8 × 100 epochs ~ 2-4 hours on A100)
#
# Usage:
#   sbatch scripts/slurm_sadhwani.sh
#   sbatch scripts/slurm_sadhwani.sh --export=EPOCHS=200,ENSEMBLE=8
# =============================================================================

set -euo pipefail

# -- Environment setup (adapt to your cluster's module system) --
# module load python/3.11 cuda/12.1 pytorch/2.3   # uncomment for module-based clusters
# source /path/to/venv/bin/activate                # uncomment for virtualenv

# Create log directory
mkdir -p logs

# Print environment info
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Date: $(date)"
echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
nvidia-smi || true
echo "============================================="

# -- Configurable parameters (override via SLURM --export or env vars) --
EPOCHS=${EPOCHS:-200}
ENSEMBLE=${ENSEMBLE:-8}
BATCH_SIZE=${BATCH_SIZE:-8192}
LR=${LR:-0.1}
DROPOUT=${DROPOUT:-0.5}
PATIENCE=${PATIENCE:-15}
SEED=${SEED:-42}
DEPTH_COMPARISON=${DEPTH_COMPARISON:-1}

# -- Run training --
python3 scripts/run_sadhwani_train.py \
    --device cuda \
    --n-epochs "$EPOCHS" \
    --n-ensemble "$ENSEMBLE" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --dropout "$DROPOUT" \
    --patience "$PATIENCE" \
    --seed "$SEED" \
    --hidden-sizes 200 140 140 140 140 \
    --eval-times 24 48 72 \
    $([ "$DEPTH_COMPARISON" = "1" ] && echo "--depth-comparison" || true)

echo ""
echo "=== Job completed at $(date) ==="
