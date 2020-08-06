#!/bin/sh


#BSUB -J deeplab_cpu
#BSUB -q workq
#BSUB -o deeplab_%J.out
#BSUB -e deeplab_%J.err
##BSUB -n 48
#BSUB -R "10*{select[clx2s8260L] span[ptile=4]}"
#BSUB -W 48:00
#BSUB -P O:Description


source /opt/opt/intel/impi/2019.4.243/parallel_studio_xe_2019/bin/psxevars.sh
source /opt/opt/intel/compiler/2019u4/bin/compilervars.sh intel64

source /panfs/users/Xrhekst/virtualenvs/EXA_TF2_HORO/bin/activate
# pip install -I tensorflow-2.1.0-cp37-cp37m-manylinux2010_x86_64.whl -f ./ --no-index --no-cache-dir --user
# CC=mpiicc CXX=mpiicpc pip install -I horovod-0.19.0.tar.gz -f ./ --no-index --no-cache-dir --user


# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"
export LD_LIBRARY_PATH=/opt/intel/python/3.7_2020/lib:$LD_LIBRARY_PATH #/opt/crtdc/mvapich2/2.2-intel/lib:$LD_LIBRARY_PATH
export PATH=/opt/intel/python/3.7_2020/bin:$PATH

TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" mpiexec -map-by ppr:2:node -np 20 --bind-to socket python -u train.py \
--no_cuda \
--horovod \
--img_size 2048 \
--log_dir "/panfs/users/Xrhekst/cartesius/deeplab/CAMELYON_TRAINING/logs" \
--fp16_allreduce \
--train_centers 0 \
--val_centers 1 2 3 4 \
--val_split 0.15 \
--train_path '/panfs/users/Xrhekst/files/examode/CAMELYON17/center_XX' \
--valid_path '/panfs/users/Xrhekst/files/examode/CAMELYON17/center_XX' \
--weights_path '/panfs/users/Xrhekst/files/examode/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' \
--log_every 4 \
--validate_every 8