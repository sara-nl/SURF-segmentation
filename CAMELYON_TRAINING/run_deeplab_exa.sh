#!/bin/bash
#SBATCH -N 2
#SBATCH -t 8:00:00
#SBATCH -p gpu_titanrtx
clear
module purge
module load 2019
module load Python/3.6.6-foss-2018b
module load CUDA/10.0.130
module load cuDNN/7.6.3-CUDA-10.0.130
module load NCCL/2.4.7-CUDA-10.0.130
#module load mpi/openmpi/3.1.2-cuda10
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

conda deactivate

# Creating virtualenv
VIRTENV=EXA_GPU_TF2_HOROVOD_GPU
VIRTENV_ROOT=~/.virtualenvs

if [ ! -z $1 ] && [ $1 = 'create' ]; then
echo "Creating virtual environment $VIRTENV_ROOT/$VIRTENV"
rm -r $VIRTENV_ROOT/$VIRTENV
virtualenv $VIRTENV_ROOT/$VIRTENV
fi

# Sourcing virtualenv
echo "Sourcing virtual environment $VIRTENV_ROOT/$VIRTENV/bin/activate"
source $VIRTENV_ROOT/$VIRTENV/bin/activate

if [ ! -z $1 ] && [ $1 = 'create' ]; then
pip3 install -r requirements.txt
fi

# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"

# Tensorflow
echo "Installing Tensorflow"
pip3 install tensorflow-gpu --no-cache-dir

# Horovod
echo "Installing Horovod"

export HOROVOD_WITH_TENSORFLOW=1
pip3 install horovod --no-cache-dir

echo "Performing Training..."
# python train.py --img_size 2048 --train_centers 1 2 3 4 --val_centers 1 2 3 4 --batch_size 32 --no_cuda --horovod
# mpirun -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python train.py --img_size 256 --train_centers 1 2 3 --val_centers 4 --horovod --batch_size 2
mpirun -map-by ppr:4:node -np 8 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python train.py --img_size 1024 --dataset 17 --horovod --batch_size 2



