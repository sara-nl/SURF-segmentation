#!/bin/bash



module purge
module use ~/environment-modules-lisa
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243


export PATH=/home/$USER/examode/lib_deps/bin:$PATH
export LD_LIBRARY_PATH=/home/$USER/examode/lib_deps/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/$USER/examode/lib_deps/lib:$LD_LIBRARY_PATH
export CPATH=/home/$USER/examode/lib_deps/include:$CPATH

# SOURCE VIRTUAL ENVIRONMENT WITH OpenSlide
source ~/virtualenvs/openslide/bin/activate


python -u preprocess_data_par.py \
--patch_size 256 \
--num_threads 1 \
--save_neg True \
--save_png True \
--train_tumor_wsi_path /nfs/examode/Colon/AOEC \
--save_tumor_negative_path AOEC \
--format svs




