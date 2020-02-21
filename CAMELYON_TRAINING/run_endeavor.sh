#!/bin/sh


#BSUB -J deeplab_cpu
#BSUB -q workq
#BSUB -o deeplab_%J.out
#BSUB -e deeplab_%J.err
##BSUB -n 48
#BSUB -R "50*{select[clx2s8260L] span[ptile=1]}"
#BSUB -W 48:00
#BSUB -P O:Description


#source /opt/opt/intel/impi/2019.4.243/parallel_studio_xe_2019/bin/psxevars.sh
#source /opt/opt/intel/compiler/2019u4/bin/compilervars.sh intel64
# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"
export LD_LIBRARY_PATH=/opt/opt/intel/python/3.7_2020/lib:$LD_LIBRARY_PATH #/opt/crtdc/mvapich2/2.2-intel/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/opt/opt/intel/python/3.7_2020/lib:$LIBRARY_PATH

source /panfs/users/Xrhekst/cartesius/virtualenvs/EXA_TF_HOROVOD/bin/activate

XLA_FLAGS=--xla_hlo_profile TF_XLA_FLAGS=--tf_xla_cpu_global_jit mpirun -map-by ppr:1:node -np 1 --bind-to socket python -u train.py \
--no_cuda \
--fp16_allreduce \
--train_path '/panfs/users/Xrhekst/files/examode/CAMELYON16' \
--valid_path '/panfs/users/Xrhekst/files/examode/CAMELYON16'
