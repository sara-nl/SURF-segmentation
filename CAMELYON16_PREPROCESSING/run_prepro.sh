#!/bin/bash


source ~/.virtualenvs/openslide/bin/activate


python -u /home/rubenh/projects/camelyon/deeplab/CAMELYON16_PREPROCESSING/preprocess_data_par.py \
--patch_size 1024 \
--num_threads 4 \
--save_png True \





