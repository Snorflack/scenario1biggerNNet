#!/bin/bash
#SBATCH --output=/home/joseph.coble/Thesis/Thesis/scenario1/Coble_results/alphazero_gameScenario1-%j.txt
#SBATCH --time=0
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=test
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END
#SBATCH --mail-user=Joseph.coble@nps.edu







python3 server.py Scenario1middleIsland.scn --redAI pass-agg --blueAI alphazero --blueNeuralNet /home/joseph.coble/Thesis/Thesis/scenario1/server/azg/temp --nReps 20 --blueReplay test.js