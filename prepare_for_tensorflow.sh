#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:p100:1
#SBATCH -t 12:00:00

set -x

#move to working directory
# this job assumes:
# - all input data is stored in this directory 
# - all output should be stored in this directory
cd $SCRATCH/research/data/wikipedia

# Run the program
python2 ./build_wikipedia_data.py --text_dir="./text/AA/" --output_dir="./" --word_counts_output_file="./word_counts.txt"
