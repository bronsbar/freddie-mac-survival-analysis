#!/usr/bin/env bash

# =============================================================================
# Joint Model — SLURM submission for VSC Genius cluster
#
# Bayesian MCMC (Pyro NUTS) for joint longitudinal + competing risks model.
# CPU-only: float64 required for NUTS numerical stability.
#
# Usage:
#   sbatch scripts/slurm_joint_model.sh
#
# Override defaults via --export:
#   sbatch --export=MAX_LOANS=5000,NUM_CHAINS=2 scripts/slurm_joint_model.sh
# =============================================================================

#SBATCH --job-name="joint-model"
#SBATCH --nodes="1"
#SBATCH --ntasks="1"
#SBATCH --cpus-per-task="9"
#SBATCH --cluster="genius"
#SBATCH --mem="44G"
#SBATCH --time="12:00:00"
#SBATCH --partition="batch_long"
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
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import pyro; print(f'Pyro    {pyro.__version__}')"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Memory:     $SLURM_MEM_PER_NODE MB"
echo "============================================="

# ── Configurable parameters (override via SLURM --export) ───────────────────
NUM_CHAINS=${NUM_CHAINS:-4}
NUM_SAMPLES=${NUM_SAMPLES:-2000}
NUM_WARMUP=${NUM_WARMUP:-1000}
TARGET_ACCEPT=${TARGET_ACCEPT:-0.90}
MAX_LOANS=${MAX_LOANS:-10000}
MAX_TEST_LOANS=${MAX_TEST_LOANS:-10000}
N_INTERIOR_KNOTS=${N_INTERIOR_KNOTS:-3}
N_POSTERIOR_CIF=${N_POSTERIOR_CIF:-200}
SEED=${SEED:-42}

# ── Run ──────────────────────────────────────────────────────────────────────
python run_joint_model.py \
    --num-chains "$NUM_CHAINS" \
    --num-samples "$NUM_SAMPLES" \
    --num-warmup "$NUM_WARMUP" \
    --target-accept "$TARGET_ACCEPT" \
    --max-loans "$MAX_LOANS" \
    --max-test-loans "$MAX_TEST_LOANS" \
    --n-interior-knots "$N_INTERIOR_KNOTS" \
    --n-posterior-cif "$N_POSTERIOR_CIF" \
    --eval-times 24 48 72 \
    --seed "$SEED"

echo ""
echo "=== Job completed at $(date) ==="
