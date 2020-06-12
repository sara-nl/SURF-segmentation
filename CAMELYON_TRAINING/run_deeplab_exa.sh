#!/bin/bash
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH -p gpu_short
clear
module use ~/environment-modules-lisa
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243
module list

# Setting ENV variables
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_NCCL_HOME=$EBROOTNCCL
export HOROVOD_GPU_ALLREDUCE=NCCL
#export HOROVOD_GPU_ALLGATHER=MPI
#export HOROVOD_GPU_BROADCAST=MPI


# Creating virtualenv
#VIRTENV=EXA_GPU_TF2_HOROVOD_GPU
#VIRTENV_ROOT=~/virtualenvs

#if [ ! -z $1 ] && [ $1 = 'create' ]; then
#echo "Creating virtual environment $VIRTENV_ROOT/$VIRTENV"
#rm -r $VIRTENV_ROOT/$VIRTENV
#virtualenv $VIRTENV_ROOT/$VIRTENV
#fi

# Sourcing virtualenv
#echo "Sourcing virtual environment $VIRTENV_ROOT/$VIRTENV/bin/activate"
#source $VIRTENV_ROOT/$VIRTENV/bin/activate
#
#if [ ! -z $1 ] && [ $1 = 'create' ]; then
#pip3 install -r requirements.txt
#fi

# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"

# Tensorflow
#echo "Installing Tensorflow"
#pip install tensorflow


# Horovod
#echo "Installing Horovod"

#export HOROVOD_WITH_TENSORFLOW=1
#pip install horovod
pip install scikit-learn --user
pip install Pillow --user
pip install tqdm --user


echo "Performing Training..."
# python train.py --img_size 2048 --train_centers 1 2 3 4 --val_centers 1 2 3 4 --batch_size 32 --no_cuda --horovod
# mpirun -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python train.py --img_size 256 --train_centers 1 2 3 --val_centers 4 --horovod --batch_size 2
mpirun -map-by ppr:1:node -np 1 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python -u train.py \
--img_size 1024 \
--dataset 17 \
--horovod \
--batch_size 1 \
--fp16_allreduce \
--train_centers 2 3 4 \
--val_centers 1 \
--log_dir /home/rubenh/examode/deeplab/CAMELYON_TRAINING/logs/train_data/ \
--log_every 2 \
--num_steps 20
