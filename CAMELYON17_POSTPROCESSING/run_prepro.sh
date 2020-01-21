#!/bin/bash


source ~/virtualenvs/openslide/bin/activate


python /home/rubenh/examode/deeplab/CAMELYON17_POSTPROCESSING/preprocess_data_par.py \
--patch_size 2048 \
--num_threads 4 \
--save_png True \








