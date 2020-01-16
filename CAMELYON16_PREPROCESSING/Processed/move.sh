#!/bin/bash


#SBATCH -t 2:00:00
#SBATCH -p broadwell

mv /home/rubenh/projects/deeplab/CAMELYON16_PREPROCESSING/Processed/patch-based-classification_224 /nfs/managed_datasets/CAMELYON16/
mv /home/rubenh/projects/deeplab/CAMELYON16_PREPROCESSING/Processed/patch-based-classification_704 /nfs/managed_datasets/CAMELYON16/
mv /home/rubenh/projects/deeplab/CAMELYON16_PREPROCESSING/Processed/patch-based-classification_1024 /nfs/managed_datasets/CAMELYON16/




