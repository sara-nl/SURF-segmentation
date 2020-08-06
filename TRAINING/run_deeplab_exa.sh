#!/bin/bash
#SBATCH -N 6
#SBATCH -t 48:00:00
#SBATCH -p gpu_titanrtx
np=$(($SLURM_NNODES * 4))

module purge
module load 2019
module load Python/3.6.6-foss-2019b
module load cuDNN/7.6.5.32-CUDA-10.1.243
source ~/virtualenvs/openslide/bin/activate
pip install tensorflow==2.3.0
pip install scikit-learn 
pip install Pillow 
pip install tqdm 
pip install six
pip install opencv-python
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

mpirun -map-by ppr:4:node -np 8 -x LD_LIBRARY_PATH -x PATH python -u train.py \
--img_size 1024 \
--horovod \
--model effdet \
--batch_size 1 \
--fp16_allreduce \
--log_dir /home/rubenh/SURF-deeplab/TRAINING/logs/test/ \
--log_every 2 \
--num_steps 100000 \
--slide_format tif \
--label_format xml \
--valid_slide_format tif \
--valid_label_format xml \
--slide_path /nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor \
--label_path /nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/XML \
--valid_slide_path /nfs/managed_datasets/CAMELYON16/Testset/Images \
--valid_label_path /nfs/managed_datasets/CAMELYON16/Testset/Ground_Truth/Annotations \
--data_sampler radboud \
--label_map _0:1 _2:2 \
--sample_processes 1 \
--validate_every 4096

exit