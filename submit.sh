#!/bin/bash
#SBATCH --job-name=benchmark_cd
#SBATCH --partition=internal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/benchmark_cd_%A-%a.log
#SBATCH --error=logs/benchmark_cd_%A-%a.log
#SBATCH --array=0-107
# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid
# ─────────────────────────────────────────────────────────────────────────────
N_VALUES=(40 50 60 80) #=(4 6 8 10)
DENSITY_VALUES=(0.5 0.7 0.9)
T_VALUES=(500 1000 2000)
NFOURIER_VALUES=(0 1 2)

# Compute total number of combinations and set the array size accordingly:
# #SBATCH --array=0-107   (4 * 3 * 3 * 3 - 1 = 107)


# ─────────────────────────────────────────────────────────────────────────────
# Decode SLURM_ARRAY_TASK_ID into the 4 parameters
# ─────────────────────────────────────────────────────────────────────────────
N_LEN=${#N_VALUES[@]}
D_LEN=${#DENSITY_VALUES[@]}
T_LEN=${#T_VALUES[@]}
NF_LEN=${#NFOURIER_VALUES[@]}

TASK=${SLURM_ARRAY_TASK_ID}

NF_IDX=$(( TASK % NF_LEN ))
TASK=$(( TASK / NF_LEN ))
T_IDX=$(( TASK % T_LEN ))
TASK=$(( TASK / T_LEN ))
D_IDX=$(( TASK % D_LEN ))
TASK=$(( TASK / D_LEN ))
N_IDX=$(( TASK % N_LEN ))

N=${N_VALUES[$N_IDX]}
DENSITY=${DENSITY_VALUES[$D_IDX]}
T=${T_VALUES[$T_IDX]}
NFOURIER=${NFOURIER_VALUES[$NF_IDX]}

echo "Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  N=${N}  density=${DENSITY}  T=${T}  nfourier=${NFOURIER}"

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p logs
source .venv/bin/activate

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
python3 benchmark_cd.py "${N}" "${DENSITY}" "${T}" "${NFOURIER}"

echo "Done — $(date '+%Y-%m-%d %H:%M:%S')"