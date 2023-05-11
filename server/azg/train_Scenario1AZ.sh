#!/bin/bash
#SBATCH --output=/home/joseph.coble/Thesis/Thesis/scenario1biggerNNet/Coble_results/alphazero_gameScenario1-test-%j.txt
#SBATCH --time=0
#SBATCH --cpus-per-task=6
#SBATCH --job-name=A1GBiggerNNet
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=barton
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=Joseph.coble@nps.edu


python3 alphazero_main.py 


#pip install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40 to fix issue of stable-baselines not working getting 
#Building wheels for collected packages: gym Building wheel for gym (setup.py) ... error error: subprocess-exited-with-error