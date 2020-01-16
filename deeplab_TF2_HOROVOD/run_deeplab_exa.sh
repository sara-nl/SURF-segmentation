#!/bin/bash

#SBATCH -p gpu_titanrtx
#SBATCH -t 8:00:00
#SBATCH -N 8


module purge
module load 2019
module load Python/3.6.6-foss-2018b
module load CUDA/10.0.130
module load cuDNN/7.6.3-CUDA-10.0.130
module load NCCL/2.4.7-CUDA-10.0.130
module unload GCC
module unload GCCcore
module unload binutils
module unload zlib
module load GCC/8.2.0-2.31.1
module unload compilerwrappers

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
VIRTENV=EXA_GPU_TF2_HORO_GPU
VIRTENV_ROOT=~/.virtualenvs

#rm -rf $VIRTENV_ROOT/$VIRTENV

#echo "Creating virtual environment $VIRTENV_ROOT/$VIRTENV --system-site-packages"
#virtualenv $VIRTENV_ROOT/$VIRTENV --system-site-packages

# Sourcing virtualenv
echo "Sourcing virtual environment $VIRTENV_ROOT/$VIRTENV/bin/activate"
source $VIRTENV_ROOT/$VIRTENV/bin/activate

# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"

# Tensorflow
echo "Installing Tensorflow"
pip3 install tensorflow-gpu --no-cache-dir

# Horovod
echo "Installing Horovod"
pip3 install horovod --no-cache-dir

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="/home/rubenh/projects/deeplab/models/research/deeplab/deeplab_TF2_HOROVOD"

echo "Performing Training..."


mpirun -map-by ppr:4:node -np 32 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib --bind-to socket python -u train.py --img_size 2048 --cam 17




