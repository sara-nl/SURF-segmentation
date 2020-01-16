#!/bin/bash
#SBATCH -t 8:00:00
source ~/.virtualenvs/openslide/bin/activate



pip install tensorflow==1.5.1
pip install Pillow --user
pip install opencv-python
pip install openslide-python
pip install imageio
pip install scikit-image

echo "Starting Script"

python Stain_Color_Normalization.py \
	--mode=png_prediction \
	--logs_dir=logs_train/ \
	--data_dir=/home/rubenh/projects/deeplab/CAMELYON16_PREPROCESSING/Processed/patch-based-classification_1024/raw-data/train/label-1/ \
	--tmpl_dir=/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor/ \
	--out_dir=/home/rubenh/projects/deeplab/CAMELYON16_PREPROCESSING/Histopathology-Stain-Color-Normalization/1024_out/train/

python Stain_Color_Normalization.py \
	--mode=png_prediction \
	--logs_dir=logs_train/ \
	--data_dir=/home/rubenh/projects/deeplab/CAMELYON16_PREPROCESSING/Processed/patch-based-classification_1024/raw-data/validation/label-1/ \
	--tmpl_dir=/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor/ \
	--out_dir=/home/rubenh/projects/deeplab/CAMELYON16_PREPROCESSING/Histopathology-Stain-Color-Normalization/1024_out/validation/




echo "Finished"
