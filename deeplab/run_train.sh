#!/bin/bash
#SBATCH -N 3
#SBATCH -t 16:00:00
#SBATCH -p gpu_titanrtx
#SBATCH -o R-cam16.out
#SBATCH -e R-cam16.err


np=$(($SLURM_NNODES * 4))


VENV_NAME=openslide-py38
module purge
module load 2019
module load 2020
module load Python/3.8.2-GCCcore-9.3.0
module load OpenMPI/4.0.3-GCC-9.3.0
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module unload GCCcore
module unload ncurses
module load CMake/3.11.4-GCCcore-8.3.0
source ~/virtualenvs/$VENV_NAME/bin/activate
#pip install --force-reinstallll tensorflow==2.3.0
#pip install tensorboard==2.3.0
pip install scikit-learn 
pip install Pillow 
pip install tqdm 
pip install six
pip install opencv-python
pip install openslide-python
pip install pandas
pip install numba
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_WITHOUT_GLOO=1
export HOROVOD_GPU_BROADCAST=NCCL
export HOROVOD_WITH_TENSORFLOW=1
export PATH=/home/$USER/virtualenvs/$VENV_NAME/bin:$PATH
export LD_LIBRARY_PATH=/home/$USER/virtualenvs/$VENV_NAME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/$USER/virtualenvs/$VENV_NAME/lib:$LD_LIBRARY_PATH
export CPATH=/home/$USER/virtualenvs/$VENV_NAME/include:$CPATH
export HOROVOD_NCCL_HOME=$EBROOTNCCL
# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"
export TF_GPU_THREAD_MODE=gpu_private
hosts=`sh ~/hosts.sh`


horovodrun -np 1 \
--mpi-args="--map-by ppr:4:node" \
--hosts $hosts \
python -u train.py \
--image_size 2048 \
--batch_size 1 \
--verbose debug \
--fp16_allreduce \
--log_dir /home/rubenh/SURF-segmentation/deeplab/logs/ \
--log_every 1 \
--min_lr 0.0 \
--max_lr 0.001 \
--validate_every 240 \
--steps_per_epoch 50000 \
--slide_path '/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor' \
--label_path '/nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/XML' \
--valid_slide_path '/home/rubenh/SURF-segmentation/efficientdet/keras/trainwsi' \
--valid_label_path '/home/rubenh/SURF-segmentation/efficientdet/keras/trainwsi' \
--bb_downsample 7 \
--batch_tumor_ratio 1 \
--optimizer Adam \
--lr_scheduler cyclic

exit
