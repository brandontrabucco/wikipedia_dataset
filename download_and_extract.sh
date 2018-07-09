#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:p100:1
#SBATCH -t 5:00:00

set -x

#move to working directory
# this job assumes:
# - all input data is stored in this directory 
# - all output should be stored in this directory
cd $SCRATCH/research/data/wikipedia

# Run the program
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
bzip2 -d enwiki-latest-pages-articles.xml.bz2
