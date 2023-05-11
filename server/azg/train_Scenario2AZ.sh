#!/bin/bash
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=NS1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/joseph.coble/Thesis/Thesis/atlatl-coble/Coble_results/alphazero_gameScenario2-out-%j.txt
#SBATCH --time=0


python3 alphazero_main.py \
--log_dir="/home/joseph.coble/Thesis/Thesis/atlatl-coble/logs/$(echo $USER)_$(date +%Y-%m-%d_%H-%M-%S-%N)_alphazero_gameScenario2/" 


