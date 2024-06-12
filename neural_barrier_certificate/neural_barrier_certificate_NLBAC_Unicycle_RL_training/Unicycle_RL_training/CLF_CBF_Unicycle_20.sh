#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --time=11:59:00
#SBATCH --job-name=short_MBPPO_LAG_Pvtol_02
#SBATCH --mail-type=all
#SBATCH --mail-user=liqun.zhao@wolfson.ox.ac.uk

module purge
module load Anaconda3/2021.11
source activate mujoco

python main.py --env Unicycle --gamma_b 50 --max_episodes 200  --cuda --updates_per_step 2 --batch_size 128 --seed 39  --start_steps 1000