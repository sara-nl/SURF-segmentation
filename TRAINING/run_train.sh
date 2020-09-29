#!/bin/bash
#SBATCH -N 2
#SBATCH -t 14:00:00
#SBATCH -p gpu_titanrtx
#SBATCH -o R-256tnode1dip.out
#SBATCH -e R-256tnode1dip.err
#SBATCH -w r28n1,r28n2



np=$(($SLURM_NNODES * 4))


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
source ~/virtualenvs/openslide-py38/bin/activate
pip install tensorflow==2.3.0
#pip install tensorboard==2.3.0
pip install scikit-learn 
pip install Pillow 
pip install tqdm 
pip install six
pip install opencv-python
pip install openslide-python
pip install pyvips
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_GPU_ALLREDUCE=NCCL
#export HOROVOD_WITH_MPI=1
#export HOROVOD_WITHOUT_MPI=1
export HOROVOD_WITHOUT_GLOO=1
export HOROVOD_GPU_BROADCAST=NCCL
export HOROVOD_WITH_TENSORFLOW=1
export PATH=/home/$USER/virtualenvs/openslide-py38/bin:$PATH
export LD_LIBRARY_PATH=/home/$USER/virtualenvs/openslide-py38/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/$USER/virtualenvs/openslide-py38/lib:$LD_LIBRARY_PATH
export CPATH=/home/$USER/virtualenvs/openslide-py38/include:$CPATH
# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"
export TF_GPU_THREAD_MODE=gpu_private

# r29n4:4,r29n5:4,r33n6:4,r34n1:4,r34n2:4,r34n3:4,r34n4:4,r34n5:4
#--mpi-args="--map-by ppr:4:node" \
#--evaluate \
#--model_dir /home/rubenh/SURF-deeplab/TRAINING/logs/256twonode/saved_model \

horovodrun -np 8 \
--mpi-args="--map-by ppr:4:node" \
--hosts r28n1:4,r28n2:4 \
python -u train.py \
--img_size 256 \
--horovod \
--model deeplab \
--batch_size 4 \
--fp16_allreduce \
--log_dir /home/rubenh/SURF-deeplab/TRAINING/logs/256tnode1dip/ \
--log_every 2 \
--validate_every 5000 \
--num_steps 50000 \
--slide_path /nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor \
--label_path /nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/Mask \
--valid_slide_path /nfs/managed_datasets/CAMELYON16/Testset/Images \
--valid_label_path /nfs/managed_datasets/CAMELYON16/Testset/Ground_Truth/Masks \
--bb_downsample 7 \
--data_sampler surf \
--batch_tumor_ratio 1

exit

horovodrun -np 4 \
--mpi-args="--map-by ppr:4:node" \
--hosts r28n1:4,r28n2:4 \
python -u train.py \
--img_size 512 \
--horovod \
--model deeplab \
--batch_size 2 \
--fp16_allreduce \
--log_dir /home/rubenh/SURF-deeplab/TRAINING/logs/512snode1dip/ \
--log_every 2 \
--validate_every 5000 \
--num_steps 50000 \
--slide_path /nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor \
--label_path /nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/Mask \
--valid_slide_path /nfs/managed_datasets/CAMELYON16/Testset/Images \
--valid_label_path /nfs/managed_datasets/CAMELYON16/Testset/Ground_Truth/Masks \
--bb_downsample 7 \
--data_sampler surf \
--batch_tumor_ratio 1


exit









module purge
module load 2019
module load Python/3.6.6-foss-2019b
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module load CMake/3.12.1-GCCcore-8.3.0
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
export HOROVOD_GPU_BROADCAST=NCCL
export PATH=/home/$USER/virtualenvs/openslide/bin:$PATH
export LD_LIBRARY_PATH=/home/$USER/virtualenvs/openslide/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/$USER/virtualenvs/openslide/lib:$LD_LIBRARY_PATH
export CPATH=/home/$USER/virtualenvs/openslide/include:$CPATH
# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"
pip uninstall horovod
pip install --no-cache-dir horovod[tensorflow]

#mpirun -map-by ppr:4:node -np 8 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH python -u train.py \
#--mpi-args="--map-by ppr:4:node" \
#--hierarchical-allreduce \
#--autotune \
#--hosts r35n3:4,r35n4:4,r35n5:4,r36n1:4,r36n3:4,r36n4:4 \

horovodrun -np 4 --verbose \
--mpi-args="--map-by ppr:4:node" \
--hosts r31n5:4 \
python -u train.py \
--img_size 512 \
--horovod \
--model effdetd0 \
--batch_size 1 \
--fp16_allreduce \
--log_dir /home/rubenh/SURF-deeplab/TRAINING/logs/2048_conv_6nodes_no_optim/ \
--log_every 2 \
--validate_every 4 \
--num_steps 5000 \
--slide_format tif \
--label_format xml \
--valid_slide_format tif \
--valid_label_format xml \
--slide_path /nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor \
--label_path /nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/XML \
--valid_slide_path /nfs/managed_datasets/CAMELYON16/Testset/Images \
--valid_label_path /nfs/managed_datasets/CAMELYON16/Testset/Ground_Truth/Annotations \
--data_sampler radboud \
--label_map _0:1 _2:2

exit


