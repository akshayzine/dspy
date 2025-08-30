#!/bin/bash
#SBATCH --job-name=dspy-gpu_test
#SBATCH --partition=gpu              # gpu queue on Avon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # one Python process
#SBATCH --cpus-per-task=10           # CPU threads for Polars/BLAS
#SBATCH --gres=gpu:quadro_rtx_6000:1    # request 1 Quadro RTX 6000
#SBATCH --mem-per-cpu=3700           # 3.7 GB × 16 ≈ 59 GB total
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# (Optional) clean modules; you’re using a venv
module purge 2>/dev/null

# Activate your virtualenv
module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.1
source "$HOME/venvs/dspy312/bin/activate"

# Run from your repo’s run folder
cd "$HOME/server_codes/dspy/run"
python run_simulation.py
