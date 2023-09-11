#!/bin/sh -l
#SBATCH --account=abuganza
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --job-name=train_murine
#SBATCH --mail-user=vtac@purdue.edu
#SBATCH --mail-type=NONE
#SBATCH --cpus-per-task=20
#SBATCH -o "slurm_murine.out"

module load anaconda
module load abaqus/2020
source activate jax

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

python -u train_murine.py
cd abaqus
abaqus job=PU1_034_3.inp user=NODEMM interactive ask_delete=off cpus=10
