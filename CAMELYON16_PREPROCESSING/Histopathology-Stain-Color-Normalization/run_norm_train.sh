#!/bin/bash

##SBATCH -p gpu
##SBATCH -t 3:00:00



source ~/.virtualenvs/openslide/bin/activate



pip install tensorflow==1.5.1
pip install Pillow --user
pip install opencv-python
pip install openslide-python
pip install imageio
pip install scikit-image

echo "Starting Script"


rm -rf logs_train/*
rm -rf logs_DCGMM_HSD/*

python Stain_Color_Normalization.py \
	--mode=train \
	--logs_dir=logs_train/ \
	--data_dir=/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor \
	--tmpl_dir=/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor/Tumor_001 \
	--out_dir=normal_out/



echo "Finished"
