#!/bin/bash


module load 2019
module load Python/3.7.5-foss-2018b

source ~/.virtualenvs/openslide/bin/activate


python /home/rubenh/projects/camelyon/deeplab/CAMELYON17_PREPROCESSING/preprocess_data_par.py \
--patch_size 1024 \
--num_threads 4 \
--save_png True \
--proc_center 1






