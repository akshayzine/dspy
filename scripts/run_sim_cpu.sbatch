#!/bin/bash
#SBATCH --job-name=dspy-cpu_test
#SBATCH --partition=compute          # CPU queue on Avon (or omit to use default)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3700
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out


# (Optional) clean modules; you’re using a venv
module purge 2>/dev/null

# Activate your virtualenv
module load GCCcore/13.3.0 Python/3.12.3
source "$HOME/venvs/dspy312/bin/activate"

# Run from your repo’s run folder
cd "$HOME/server_codes/dspy/run"
python run_simulation.py