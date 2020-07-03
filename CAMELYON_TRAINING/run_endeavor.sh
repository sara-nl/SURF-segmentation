#!/bin/sh

#BSUB -J deeplab_cpu
#BSUB -q workq
#BSUB -o deeplab_%J.out
#BSUB -e deeplab_%J.err
#BSUB -R "64*{select[clx2s8260hdr] span[ptile=2]}"
#BSUB -W 48:00
#BSUB -P O:Description

source /opt/opt/intel/impi/2019.7.217/parallel_studio_xe_2020/bin/psxevars.sh
source /opt/opt/intel/compiler/2020u1/bin/compilervars.sh intel64

source /panfs/users/Xvcodre/tf_2.1_py3_env/bin/activate

export LD_LIBRARY_PATH=/nfs/work04/Xvcodre/sw/arch/RedHatEnterpriseServer7/EB_production/2020/software/GCCcore/8.3.0/lib64/:/nfs/work04/Xvcodre/sw/arch/RedHatEnterpriseServer7/EB_production/2020/software/GCCcore/8.3.0/lib:$LD_LIBRARY_PATH
export PATH=/nfs/work04/Xvcodre/sw/arch/RedHatEnterpriseServer7/EB_production/2020/software/GCCcore/8.3.0/bin/:$PATH

# Export MPICC
export OMP_NUM_THREADS=46
export KMP_BLOCKTIME=0
export KMP_AFFINITY='granularity=fine,verbose,compact,1,0'
export FI_PROVIDER=sockets


mpiexec.hydra -map-by ppr:1:socket -np 64 python -u train.py \
--no_cuda \
--horovod \
--img_size 2048 \
--dataset 16 \
--log_dir "logs" \
--fp16_allreduce \
--val_split 0.15 \
--weights_path '/nfs/work04/Xvcodre/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5' \
--log_every 512 \
--validate_every 4096

