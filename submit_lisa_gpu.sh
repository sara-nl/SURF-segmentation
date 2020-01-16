#!/bin/bash
##SBATCH -N 1 
##SBATCH --gres=gpu:titanrtx:2
##SBATCH -p gpu
##SBATCH -t 12:00:00
##SBATCH --gres-flags=enforce-binding 
##SBATCH -c 3

if [ ! -z $1 ]; then
    srun -t $1:00 -c 3 --gres-flags=enforce-binding --pty bash -il
fi
#############################
##                         ##
## Environment preparation ##
##                         ##
#############################
echo "===> Preparting the environment"
module load Python/3.6.3-foss-2017b NCCL/2.3.5-CUDA-10.0.130
# Add locally installed dependencies to the environment
export PATH=/home/damian/examode/camelyon/lib_deps/bin:$PATH
export LD_LIBRARY_PATH=/home/damian/examode/camelyon/lib_deps/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/damian/examode/camelyon/lib_deps/lib:$LD_LIBRARY_PATH
export CPATH=/home/damian/examode/camelyon/lib_deps/include:$CPATH
# Activate the environment that has:
source /home/damian/examode/camelyon/lisa_gpu/bin/activate
echo "===> Packages available in the virtual environment:"
pip list
# Set the Horovod environment variables
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_NCCL_HOME=$EBROOTNCCL
export HOROVOD_GPU_ALLREDUCE=NCCL

