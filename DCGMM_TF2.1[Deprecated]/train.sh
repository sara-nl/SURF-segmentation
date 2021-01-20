#!/bin/bash
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -p gpu_titanrtx

module purge
module load 2020
module load 2019
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.6.6



"""
TRAINING:


python3 main.py \
--img_size 256 \
--batch_size 16 \
--epochs 5 \
--num_clusters 2 \
--train_path /home/rubenh/examode/deeplab/CAMELYON16_PREPROCESSING/Radboudumc \
--valid_path /home/rubenh/examode/deeplab/CAMELYON16_PREPROCESSING/Radboudumc \
--legacy_conversion \
--logdir /Radboudumc-2-clusters


EVALUATION:

python3 main.py \
--img_size 256 \
--template_path /nfs/managed_datasets/CAMELYON17/training/center_1/patches_positive_256 \
--images_path /nfs/managed_datasets/CAMELYON17/training/center_2/patches_positive_256 \
--load_path ~/examode/deeplab/DCGMM_TF2.1/logs1/checkpoint_8672 \
--legacy_conversion \
--eval_mode \
--save_path saved_images

python3 main.py \
--img_size 256 \
--template_path /nfs/managed_datasets/CAMELYON17/training/center_4/patches_positive_256 \
--images_path /nfs/managed_datasets/CAMELYON17/training/center_1/patches_positive_256 \
--load_path ~/examode/deeplab/DCGMM_TF2.1/logs3/checkpoint_123136 \
--legacy_conversion \
--eval_mode \
--batch_size 32
"""


python3 main.py \
--img_size 256 \
--template_path /home/rubenh/examode/deeplab/CAMELYON16_PREPROCESSING/Radboudumc \
--images_path /home/rubenh/examode/deeplab/CAMELYON16_PREPROCESSING/AOEC \
--load_path /home/rubenh/examode/deeplab/DCGMM_TF2.1/Radboudumc/checkpoint_2000 \
--legacy_conversion \
--eval_mode \
--num_clusters 3 \
--save_path Rad_AOEC




