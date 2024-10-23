#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --job-name=ATIAConvNext
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=2
#the ammount of memory allocated
#SBATCH --mem=2000M
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=8:00:00
#Skipping many options! see man sbatch
# From here on, we can start our program
echo python --version
echo $CUDA_VISIBLE_DEVICES

python testing_python.py