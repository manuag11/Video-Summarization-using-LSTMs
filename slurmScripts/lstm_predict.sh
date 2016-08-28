#!/bin/bash

#SBATCH -A CS381V-Visual-Recogn
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --time=00:10:00 
#SBATCH --error=../logs/lstmMSEPredict10000.err 
#SBATCH --output=../logs/lstmMSEPredict10000.out
#SBATCH --job-name=EvalLstmMSE
#SBATCH --mail-user=kunal.lad@utexas.edu
#SBATCH --mail-type=all

echo job $JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "running on node $NODE_HOSTNAME"

# cd to VideoSummarization directory.
cd ..
# Train BLSTM
luajit predict.lua -model models/lstm_mse10000.t7 -output_file results/lstm_mse_predictions10000.txt 

echo "\nFinished with exit code $? at: `date`"

