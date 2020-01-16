#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh




#echo "Removing tf-records..."
#
#mv /home/rubenh/projects/deeplab/models/research/deeplab/datasets/camelyon16/tfrecord/* /home/rubenh/projects/deeplab/models/research/deeplab/CAMELYON16_PREPROCESSING/Processed/patch-based-classification_2048/tf-records/
#

#rm -rf /home/rubenh/projects/deeplab/models/research/deeplab/datasets/camelyon16/tfrecord/*
echo "Moving tf-records..."
mv /home/rubenh/projects/deeplab/models/research/deeplab/CAMELYON16_PREPROCESSING/Processed/patch-based-classification_1024_Normalized/tf-records/* /home/rubenh/projects/deeplab/models/research/deeplab/datasets/camelyon16/tfrecord/


# Exit immediately if a command exits with a non-zero status.
module unload GCCcore/*
module unload Python/*
module unload python/*


module load python/3.5.0
#module load CUDA/10.0.130
module load CUDA/9.0.176
module unload GCCcore
module unload binutils
#module load cuDNN/7.4.2-CUDA-10.0.130
module load cuDNN/7.3.1-CUDA-9.0.176
#module load NCCL/2.3.5-CUDA-10.0.130
module load NCCL/2.3.5-5-CUDA-9.0.176

echo "$PATH"
echo "$LD_LIBRARY_PATH"

#export CUDA_VISIBLE_DEVICES=0,1,2,3
nvcc -V
#pip3 uninstall tensorflow
#pip uninstall tensorflow
pip3 install tensorflow-gpu==1.12 --user
pip3 install tensorlayer --user
#pip install tensorflow==1.13.1 --user
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"



# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
#sh download_and_convert_voc2012.sh

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories for PASCAL
#PASCAL_FOLDER="pascal_voc_seg"
#EXP_FOLDER="exp/train_on_trainval_set"
#INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/init_models"
#TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
#EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
#VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
#EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
#mkdir -p "${INIT_FOLDER}"
#mkdir -p "${TRAIN_LOGDIR}"
#mkdir -p "${EVAL_LOGDIR}"
#mkdir -p "${VIS_LOGDIR}"
#mkdir -p "${EXPORT_DIR}"


## Set up working directories for CAM16
CAM16_FOLDER="camelyon16"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${CAM16_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${CAM16_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${CAM16_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${CAM16_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${CAM16_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

rm -rf "${TRAIN_LOGDIR}/"
rm -rf "${EVAL_LOGDIR}/"

echo "Finished Making Directories"
echo "${TRAIN_LOGDIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
#TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
TF_INIT_CKPT="xception_65_coco_pretrained_2018_10_02.tar.gz"
#TF_INIT_CKPT="resnet_v1_101_2018_05_04.tar.gz"

cd "${INIT_FOLDER}"
#wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
#tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

#PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord"
CAM16_DATASET="${WORK_DIR}/${DATASET_DIR}/${CAM16_FOLDER}/tfrecord"

cd "${TMPDIR}"
mkdir -p "${TMPDIR}${CAM16_DATASET}"

cp -r "${CAM16_DATASET}" "${TMPDIR}${CAM16_DATASET}"
#echo "Training of Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"
# Train 10 iterations.
NUM_ITERATIONS=50 #18750 # ~30 000 images / batch size 24 = 1250 * 15 epochs = 18750
NAS_SESSION="False"


echo "${TMPDIR}${CAM16_DATASET}"

if [ "$NAS_SESSION" != "True" ]
then

    echo "Performing Training..."

    python3 "${WORK_DIR}"/train.py \
        --num_clones=4 \
        --base_learning_rate=0.0001 \
        --learning_rate_decay_factor=0.8 \
        --weight_decay=0.00004 \
        --learning_policy="step" \
        --learning_rate_decay_step=50 \
        --dataset="camelyon16" \
        --train_split="train" \
        --model_variant="xception_65" \
        --min_scale_factor=1 \
        --max_scale_factor=1 \
        --scale_factor_step_size=1 \
        --atrous_rates=6 \
        --atrous_rates=12 \
        --atrous_rates=18 \
        --save_summaries_secs=5 \
        --save_summaries_images=True \
        --output_stride=16 \
        --decoder_output_stride=4 \
        --last_layer_gradient_multiplier=4 \
        --train_crop_size=1024 \
        --train_crop_size=1024 \
        --train_batch_size=4 \
        --training_number_of_steps="${NUM_ITERATIONS}" \
        --fine_tune_batch_norm="False" \
        --train_logdir="${TRAIN_LOGDIR}" \
        --dataset_dir="${TMPDIR}${CAM16_DATASET}/tfrecord" \
        --initialize_last_layer="False" \
        --last_layers_contain_logits_only="True" \
        --log_steps=1 \
        --hard_example_mining_step=300 \
        --top_k_percent_pixels=2 \
    	--tf_initial_checkpoint="${INIT_FOLDER}/xception_65_coco_pretrained/x65-b2u1s2p-d48-2-3x256-sc-cr300k_init.ckpt" \
        #--profile_logdir="${TRAIN_LOGDIR}"
        #--aspp_with_batch_norm=False \

    #	decay = 6250
    #	hard exmaple mining step = 18000
else

    echo "Performing Neural Architecture Search..."

    python3 "${WORK_DIR}"/train.py \
        --num_clones=4 \
        --base_learning_rate=0.0001 \
        --learning_rate_decay_factor=0.8 \
        --weight_decay=0.00004 \
        --learning_policy="step" \
        --learning_rate_decay_step=20000 \
        --dataset="camelyon16" \
        --train_split="train" \
        --model_variant="nas_hnasnet" \
        --min_scale_factor=1 \
        --max_scale_factor=1 \
        --scale_factor_step_size=1 \
        --save_summaries_secs=5 \
        --save_summaries_images=True \
        --last_layer_gradient_multiplier=6 \
        --train_crop_size=704 \
        --train_crop_size=704 \
        --train_batch_size= \
        --aspp_with_batch_norm=False \
        --training_number_of_steps="${NUM_ITERATIONS}" \
        --train_logdir="${TRAIN_LOGDIR}" \
        --dataset_dir="${TMPDIR}${CAM16_DATASET}/tfrecord" \
        --log_steps=1 \
        --hard_example_mining_step=100000 \
        --top_k_percent_pixels=1 \
        --nas_stem_output_num_conv_filters=20 \
        --drop_path_keep_prob=0.9

fi


#Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.
echo "Begin Evaluation"
python3 "${WORK_DIR}"/eval.py \
  --dataset="camelyon16" \
  --eval_split="validation" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=1024 \
  --eval_crop_size=1024 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${TMPDIR}${CAM16_DATASET}/tfrecord" \
  --max_number_of_evaluations=1


echo "Begin Visualization"
# Visualize the results.
python3 "${WORK_DIR}"/vis.py \
  --logtostderr \
  --dataset="camelyon16" \
  --vis_split="validation" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size=1024 \
  --vis_crop_size=1024 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${TMPDIR}${CAM16_DATASET}/tfrecord" \
  --max_number_of_iterations=1

## Export the trained checkpoint.
#CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
#EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"
#
#python3 "${WORK_DIR}"/export_model.py \
#  --logtostderr \
#  --checkpoint_path="${CKPT_PATH}" \
#  --export_path="${EXPORT_PATH}" \
#  --model_variant="xception_65" \
#  --atrous_rates=6 \
#  --atrous_rates=12 \
#  --atrous_rates=18 \
#  --output_stride=16 \
#  --decoder_output_stride=4 \
#  --num_classes=2 \
#  --crop_size=800 \
#  --crop_size=800 \
#  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.


echo "Finished"

