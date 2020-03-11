#!/bin/bash
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH -p gpu_titanrtx

module purge
module use ~/environment-modules-lisa
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243

conda deactivate

VIRTENV=color_norm_tf2.1
VIRTENV_ROOT=~/.virtualenvs

source $VIRTENV_ROOT/$VIRTENV/bin/activate

python3 main.py --log_every 250 --normalize_imgs --legacy_conversion