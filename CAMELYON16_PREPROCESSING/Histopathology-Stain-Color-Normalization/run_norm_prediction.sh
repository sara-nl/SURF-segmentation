#!/usr/bin/env bash



source ~/.virtualenvs/lisa_gpu/bin/activate
export LD_LIBRARY_PATH=/home/rubenh/examode/camelyon/lib_deps/lib:/home/rubenh/examode/camelyon/lib_deps/lib64:/home/damian/examode/camelyon/lib_deps/lib64:/home/rubenh/.local/easybuild/Debian9/software/cuDNN/7.4.2-CUDA-10.0.130/lib64:/home/rubenh/examode/camelyon/lib_deps/lib:/home/rubenh/examode/camelyon/lib_deps/lib64:/hpc/eb/Debian9/OpenMPI/2.1.1-GCC-6.4.0-2.28/lib:/hpc/eb/Debian9/NCCL/2.3.5-CUDA-10.0.130/lib:/hpc/eb/Debian9/CUDA/10.0.130/extras/CUPTI/lib64:/hpc/eb/Debian9/CUDA/10.0.130/lib64:/sara/sw/libgfortran/32/1/lib:/sara/sw/oldwheezy-1.0/lib:/sara/sw/torque/lib
export PATH=/home/rubenh/projects/deeplab:/home/rubenh/examode/camelyon/lisa_gpu/bin:/home/rubenh/examode/camelyon/lib_deps/bin:/home/rubenh/examode/camelyon/lib_deps/bin:/hpc/eb/Debian9/CMake/3.10.2-GCCcore-6.4.0/bin:/hpc/eb/Debian9/Python/3.6.3-foss-2017b/bin:/hpc/eb/Debian9/XZ/5.2.3-GCCcore-6.4.0/bin:/hpc/eb/Debian9/SQLite/3.20.1-GCCcore-6.4.0/bin:/hpc/eb/Debian9/Tcl/8.6.7-GCCcore-6.4.0/bin:/hpc/eb/Debian9/libreadline/7.0-GCCcore-6.4.0/bin:/hpc/eb/Debian9/ncurses/6.0-GCCcore-6.4.0/bin:/hpc/eb/Debian9/bzip2/1.0.6-GCCcore-6.4.0/bin:/hpc/eb/Debian9/FFTW/3.3.6-gompi-2017b/bin:/hpc/eb/Debian9/OpenBLAS/0.2.20-GCC-6.4.0-2.28/bin:/hpc/eb/compilerwrappers/compilers:/hpc/eb/compilerwrappers/linkers:/hpc/eb/Debian9/OpenMPI/2.1.1-GCC-6.4.0-2.28/bin:/hpc/eb/Debian9/hwloc/1.11.7-GCCcore-6.4.0/sbin:/hpc/eb/Debian9/hwloc/1.11.7-GCCcore-6.4.0/bin:/hpc/eb/Debian9/numactl/2.0.11-GCCcore-6.4.0/bin:/hpc/eb/Debian9/binutils/2.28-GCCcore-6.4.0/bin:/hpc/eb/Debian9/GCCcore/6.4.0/bin:/hpc/eb/Debian9/NCCL/2.3.5-CUDA-10.0.130:/hpc/eb/Debian9/CUDA/10.0.130:/hpc/eb/Debian9/CUDA/10.0.130/bin:/opt/slurm/sbin:/opt/slurm/bin:/hpc/eb/modules-4.0.0/bin:/usr/bin:/bin:/usr/bin/X11:/usr/games:/usr/sara/bin:/usr/local/bin

module load Python/2.7.15-foss-2018b
module load CUDA/9.0.176
module load cuDNN/7.3.1-CUDA-9.0.176
module load NCCL/2.3.5-5-CUDA-9.0.176

pip install tensorflow==1.5.1 --user
echo "Starting Script"




python Stain_Color_Normalization.py \
	--mode=prediction \
	--logs_dir=logs_train/ \
	--data_dir=/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor \
	--tmpl_dir=/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor \
	--out_dir=normal_out/



echo "Finished"
