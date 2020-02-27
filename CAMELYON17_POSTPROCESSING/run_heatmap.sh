#!/bin/bash

module purge
module use ~/environment-modules-lisa
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243
module list


export PATH=/home/$USER/examode/lib_deps/bin:$PATH
export LD_LIBRARY_PATH=/home/$USER/examode/lib_deps/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/$USER/examode/lib_deps/lib:$LD_LIBRARY_PATH
export CPATH=/home/$USER/examode/lib_deps/include:$CPATH
# for importing data_utils
export PYTHONPATH=~/examode/deeplab/CAMELYON_TRAINING:$PYTHONPATH

source ~/virtualenvs/openslide/bin/activate

#pip install openslide-python --user
#pip install opencv-python

python heatmap.py








