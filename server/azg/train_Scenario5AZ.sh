#!/bin/bash
#SBATCH --output=/home/joseph.coble/Thesis/Thesis/scenario5/Coble_results/alphazero_gameScenario5-GPU50cpuct4-out-%j.txt
#SBATCH --time=0
#SBATCH --cpus-per-task=20
#SBATCH --job-name=A5G
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6G
#SBATCH --partition=barton
#SBATCH --mail-type=END
#SBATCH --mail-user=Joseph.coble@nps.edu


python3 alphazero_main.py 


