#!/usr/bin/env bash

# =============================================================================
# NN-DTSM + APC — SLURM submission for VSC Genius cluster
#
# Neural network discrete-time survival model with APC decomposition.
# GPU-accelerated training (float32 is fine for NN, unlike MCMC).
#
# Usage:
#   sbatch scripts/slurm_nn_dtsm.sh
#
# Override defaults via --export:
#   sbatch --export=N_EPOCHS=50,N_NEURONS=16 scripts/slurm_nn_dtsm.sh
# =============================================================================

#SBATCH --job-name="nn-dtsm-apc"
#SBATCH --nodes="1"
#SBATCH --ntasks="1"
#SBATCH --cpus-per-task="8"
#SBATCH --gpus-per-node="1"
#SBATCH --cluster="genius"
#SBATCH --mem="44G"
#SBATCH --time="04:00:00"
#SBATCH --partition="gpu_p100"
#SBATCH --account="lp_verbekelab"
#SBATCH --mail-type="BEGIN,END,FAIL"
#SBATCH --mail-user="bart.bronselaer@kuleuven.be"
#SBATCH --output="/vsc-hard-mounts/leuven-data/386/vsc38657/%x.o%A"
#SBATCH --error="/vsc-hard-mounts/leuven-data/386/vsc38657/%x.e%A"

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate prepay

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

cd $VSC_DATA/freddie-mac-survival-analysis/scripts

# ── Print job info ───────────────────────────────────────────────────────────
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Date:       $(date)"
echo "Working dir: $(pwd)"
echo "Python:     $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi || true
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Memory:     $SLURM_MEM_PER_NODE MB"
echo "============================================="

# ── Configurable parameters (override via SLURM --export) ───────────────────
N_HIDDEN=${N_HIDDEN:-4}
N_NEURONS=${N_NEURONS:-8}
DROPOUT=${DROPOUT:-0.0}
N_EPOCHS=${N_EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-0.001}
PATIENCE=${PATIENCE:-10}
AR_MAX_LAG=${AR_MAX_LAG:-4}
N_MC_PATHS=${N_MC_PATHS:-500}
SEED=${SEED:-42}

# ── Run ──────────────────────────────────────────────────────────────────────
python run_nn_dtsm.py \
    --device cuda \
    --use-sampled-panel \
    --n-hidden "$N_HIDDEN" \
    --n-neurons "$N_NEURONS" \
    --dropout "$DROPOUT" \
    --n-epochs "$N_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --patience "$PATIENCE" \
    --ar-max-lag "$AR_MAX_LAG" \
    --n-mc-paths "$N_MC_PATHS" \
    --eval-times 24 48 72 \
    --seed "$SEED"

echo ""
echo "=== Job completed at $(date) ==="
