# WP3 - EXAMODE COLOR INFORMATION

Internal repository regarding WP3 of Examode project concerning the color information in Whole Slide Images

- This repository is enabling color transformation on the basis of :
- - https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization



# Setup
These steps ran on LISA / Cartesius with this module environment, where we first clone and enable 2020: 

```
cd ~
git clone https://git.ia.surfsara.nl/environment-modules/environment-modules-lisa.git
module use ~/environment-modules-lisa
```

```
Currently Loaded Modulefiles:
 1) 2020                             17) bzip2/1.0.8-GCCcore-8.3.0                               
 2) GCCcore/8.3.0                    18) ncurses/6.1-GCCcore-8.3.0                               
 3) zlib/1.2.11-GCCcore-8.3.0        19) libreadline/8.0-GCCcore-8.3.0                           
 4) binutils/2.32-GCCcore-8.3.0      20) Tcl/8.6.9-GCCcore-8.3.0                                 
 5) GCC/8.3.0                        21) SQLite/3.29.0-GCCcore-8.3.0                             
 6) numactl/2.0.12-GCCcore-8.3.0     22) GMP/6.1.2-GCCcore-8.3.0                                 
 7) XZ/5.2.4-GCCcore-8.3.0           23) libffi/3.2.1-GCCcore-8.3.0                              
 8) libxml2/2.9.9-GCCcore-8.3.0      24) Python/3.7.4-GCCcore-8.3.0                              
 9) libpciaccess/0.14-GCCcore-8.3.0  25) SciPy-bundle/2019.10-foss-2019b-Python-3.7.4            
10) hwloc/1.11.12-GCCcore-8.3.0      26) Szip/2.1.1-GCCcore-8.3.0                                
11) OpenMPI/3.1.4-GCC-8.3.0          27) HDF5/1.10.5-gompi-2019b                                 
12) OpenBLAS/0.3.7-GCC-8.3.0         28) h5py/2.10.0-foss-2019b-Python-3.7.4                     
13) gompi/2019b                      29) CUDA/10.1.243                                           
14) FFTW/3.3.8-gompi-2019b           30) cuDNN/7.6.5.32-CUDA-10.1.243                            
15) ScaLAPACK/2.0.2-gompi-2019b      31) NCCL/2.5.6-CUDA-10.1.243                                
16) foss/2019b                       32) TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243     
```
Modules loaded:
```
module load 2020
module load CMake/3.15.3-GCCcore-8.3.0
module load Python/3.7.4-GCCcore-8.3.0 (this module also loads the compilers)

```

## Dependencies
For this installation we will first make sure to have a folder ```/home/examode/``` and in here we make a folder called ```lib_deps``` where we install all the dependencies needed by the code.
```
mkdir -p $HOME/examode/lib_deps
cd $HOME/examode
```
Then add the relevant values to the environment variables:
```
export PATH=/home/$USER/examode/lib_deps/bin:$PATH
export LD_LIBRARY_PATH=/home/$USER/examode/lib_deps/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/$USER/examode/lib_deps/lib:$LD_LIBRARY_PATH
export CPATH=/home/$USER/examode/lib_deps/include:$CPATH
```

### LibTIFF
1. Download a release from the official repository and untar
```
wget http://download.osgeo.org/libtiff/tiff-4.0.10.tar.gz
tar -xvf tiff-4.0.10.tar.gz
```
2. Build and configure the LibTIFF code from the inflated folder
```
cd $HOME/examode/tiff-4.0.10
CC=gcc CXX=g++ ./configure --prefix=/home/$USER/examode/lib_deps
make -j 8
```
3. Install LibTIFF
```
make install
cd ..
```

### OpenJPEG
The official install instructions are available [here](https://github.com/uclouvain/openjpeg/blob/master/INSTALL.md).
1. Download and untar a release from the official repository
```
wget https://github.com/uclouvain/openjpeg/archive/v2.3.1.tar.gz
tar -xvf v2.3.1.tar.gz
```
2. Build the OpenJPEG repository code
```
cd $HOME/examode/openjpeg-2.3.1
mkdir -p build 
CC=gcc CXX=g++ cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=/home/$USER/examode/lib_deps \
-DBUILD_THIRDPARTY:bool=on
make -j 8

```
3. Install OpenJPEG (we already added the paths to the environment variables)
```
make install
cd ..
```

### OpenSlide
1. Download and untar a release from the official repository
```
wget https://github.com/openslide/openslide/releases/download/v3.4.1/openslide-3.4.1.tar.gz
tar -xvf openslide-3.4.1.tar.gz
```
2. Build and configure the OpenSlide code
```
cd $HOME/examode/openslide-3.4.1
CC=gcc CXX=g++ PKG_CONFIG_PATH=/home/$USER/examode/lib_deps/lib/pkgconfig ./configure --prefix=/home/$USER/examode/lib_deps
make -j 8
```
3. Install OpenSlide (we already added the paths to the environment variables)
```
make install
cd ..
```


### Setting up the Python depencies (specific to LISA GPU)
First we will load the modules needed:
```
module purge
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243 

```
Then export the paths to the local installations:
```
export PATH=/home/$USER/examode/lib_deps/bin:$PATH
export LD_LIBRARY_PATH=/home/$USER/examode/lib_deps/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/$USER/examode/lib_deps/lib:$LD_LIBRARY_PATH
export CPATH=/home/$USER/examode/lib_deps/include:$CPATH
```
This will give us access to all the dependencies. Now on a GPU node (for example login-gpu1):
```
VIRTENV=openslide
VIRTENV_ROOT=~/.virtualenvs
virtualenv $VIRTENV_ROOT/$VIRTENV --system-site-packages
source $VIRTENV_ROOT/$VIRTENV/bin/activate
```
Now export environment variables for installing Horovod w/ MPI for multiworker training:
```
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_NCCL_HOME=$EBROOTNCCL
export HOROVOD_GPU_ALLREDUCE=NCCL
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"

```
Install python packages:

```
pip3 install horovod --no-cache-dir --user

```
# Preprocessing on LISA

- To start pre-processing on e.g. CAMELYON17:
```
cd ~/examode/deeplab/CAMELYON17_PREPROCESSING
# See flags in run_prepro.sh file for options
sh run_prepro.sh

```
# Running on LISA
To start a training run on LISA with the CAMELYON16 dataset, image size 1024 and batch size 2:
```
python train.py --dataset 16 --img_size 1024 --batch_size 2
```
OR (for multi-worker)
```
mpirun -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python train.py --dataset 16 --img_size 1024 --batch_size 2
```
See the * *run_deeplab_exa.sh* * file for details

A training run on positive patches of 1024 x 1024 will converge in 2 hours on 4 TITANRTX nodes (Batch Size 2, ppr:4:node) to mIoU ~ 0.90
