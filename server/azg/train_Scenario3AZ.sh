#!/bin/bash
#SBATCH --output=/home/joseph.coble/Thesis/Thesis/atlatl-coble/Coble_results/alphazero_gameScenario3-GPU50cpuct1-out-%j.txt
#SBATCH --time=0
#SBATCH --job-name=A3G
#SBATCH --gres=gpu:1
#SBATCH --mem=16G                 # total memory (RAM) per node
#SBATCH --partition=barton
#SBATCH --mail-type=END
#SBATCH --mail-user=Joseph.coble@nps.edu


python3 alphazero_main.py 

