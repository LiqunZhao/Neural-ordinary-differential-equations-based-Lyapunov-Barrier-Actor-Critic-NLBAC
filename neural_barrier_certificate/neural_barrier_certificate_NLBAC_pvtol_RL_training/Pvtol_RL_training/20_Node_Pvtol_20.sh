#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --time=11:59:00
#SBATCH --job-name=control_affine_0.2118_Node_Pvtol_1
#SBATCH --mail-type=all
#SBATCH --mail-user=liqun.zhao@wolfson.ox.ac.uk

module purge
module load Anaconda3/2021.11
export CONPREFIX=$DATA/mujoco
source activate $CONPREFIX


python main.py --env Pvtol --gamma_b 0.8 --max_episodes 210 --cuda --updates_per_step 1 --batch_size 256 --seed 200 --start_steps 1000
