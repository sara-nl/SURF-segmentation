#!/bin/bash
#SBATCH -N 4
#SBATCH -t 12:00:00
#SBATCH -p gpu_titanrtx
#SBATCH -o R-cam16.out
#SBATCH -e R-cam16.err
#SBATCH -x r34n6
##r34n6 is broken


module purge
module load 2019
module load 2020
module load Python/3.8.2-GCCcore-9.3.0
module load OpenMPI/4.0.3-GCC-9.3.0
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module unload GCCcore
module unload ncurse
module load CMake/3.11.4-GCCcore-8.3.0
VENV_NAME=openslide-py38
cd /home/rubenh/SURF-segmentation/efficientdet/keras
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
export HOROVOD_WITHOUT_PYTORCH=1
export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_WITH_TENSORFLOW=1
export HOROVOD_GPU_BROADCAST=NCCL
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
#hosts=`sh ~/hosts.sh`

echo $PATH
export PATH=$CUDA_HOME/bin:$PATH

hosts=""
for host in $(scontrol show hostnames);
do
	hosts="$hosts$host:4,"
done
hosts="${hosts%?}"
echo "HOSTS: $hosts"



horovodrun -np 1 \
--autotune \
--autotune-log-file autotunecam16.csv \
--mpi-args="--map-by ppr:1:node" \
--hosts $hosts \
python -u /home/rubenh/SURF-segmentation/efficientdet/keras/segmentation.py \
--batch_size 1 \
--optimizer Adam \
--lr_decay_method cosine \
--name efficientdet-d4 \
--log_dir /home/rubenh/SURF-segmentation/efficientdet/keras/cosine \
--steps_per_epoch 50 \
--num_epochs 500

exit




