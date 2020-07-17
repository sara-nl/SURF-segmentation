#!/bin/bash
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH -p fat
module purge
module use ~/environment-modules-lisa
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243
module list

source ~/virtualenvs/openslide/bin/activate
pip install scikit-learn 
pip install Pillow 
pip install tqdm 
pip install six
pip install opencv-python
# Setting ENV variables
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_NCCL_HOME=$EBROOTNCCL
export HOROVOD_GPU_ALLREDUCE=NCCL
#export HOROVOD_GPU_ALLGATHER=MPI
#export HOROVOD_GPU_BROADCAST=MPI
export PATH=/home/$USER/virtualenvs/openslide/bin:$PATH
export LD_LIBRARY_PATH=/home/$USER/virtualenvs/openslide/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/$USER/virtualenvs/openslide/lib:$LD_LIBRARY_PATH
export CPATH=/home/$USER/virtualenvs/openslide/include:$CPATH
# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"

mpirun -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH python -u train.py \
--img_size 1024 \
--horovod \
--batch_size 2 \
--fp16_allreduce \
--log_dir /home/rubenh/examode/deeplab/CAMELYON_TRAINING/logs/test/ \
--log_every 2 \
--num_steps 5000 \
--slide_format tif \
--mask_format tif \
--slide_path /nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor \
--mask_path /nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/Mask \
--bb_downsample 7 \
--batch_tumor_ratio 0.5 \
--log_image_path logs/test/

exit

