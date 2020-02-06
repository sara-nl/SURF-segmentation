#!/bin/bash


module load 2019
module load Python/3.6.6-foss-2018b

export PATH=/home/$USER/examode/lib_deps/bin:$PATH
export LD_LIBRARY_PATH=/home/$USER/examode/lib_deps/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/$USER/examode/lib_deps/lib:$LD_LIBRARY_PATH
export CPATH=/home/$USER/examode/lib_deps/include:$CPATH

# SOURCE VIRTUAL ENVIRONMENT WITH OpenSlide
source ~/.virtualenvs/openslide/bin/activate


python -u preprocess_data_par.py \
--patch_size 1024 \
--num_threads 4 \
--save_png True \
--data_folder cart # lisa / cart (for managed datasets folder)





