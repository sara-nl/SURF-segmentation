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

PROJECT_DIR=$PWD
VIRTENV=SURF_deeplab
VIRTENV_ROOT=~/.virtualenvs

if [ ! -z $1 ] && [ $1 = 'create' ]; then
  echo "Creating virtual environment $VIRTENV_ROOT/$VIRTENV"
  yes | rm -r $VIRTENV_ROOT/$VIRTENV
  python3 -m venv $VIRTENV_ROOT/$VIRTENV
fi

export PATH=$VIRTENV_ROOT/$VIRTENV/bin:$PATH
export LD_LIBRARY_PATH=$VIRTENV_ROOT/$VIRTENV/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$VIRTENV_ROOT/$VIRTENV/lib:$LD_LIBRARY_PATH
export CPATH=$VIRTENV_ROOT/$VIRTENV/include:$CPATH

if [ ! -z $1 ] && [ $1 = 'create' ]; then
  # INSTALLING LIBTIFF
  cd $VIRTENV_ROOT/$VIRTENV

  wget http://download.osgeo.org/libtiff/tiff-4.0.10.tar.gz
  tar -xvf tiff-4.0.10.tar.gz

  cd $VIRTENV_ROOT/$VIRTENV/tiff-4.0.10
  CC=gcc CXX=g++ ./configure --prefix=$VIRTENV_ROOT/$VIRTENV
  make -j 8
  make install

  # INSTALLING OPENJPEG
  cd $VIRTENV_ROOT/$VIRTENV

  wget https://github.com/uclouvain/openjpeg/archive/v2.3.1.tar.gz
  tar -xvf v2.3.1.tar.gz

  cd $VIRTENV_ROOT/$VIRTENV/openjpeg-2.3.1
  CC=gcc CXX=g++ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$VIRTENV_ROOT/$VIRTENV -DBUILD_THIRDPARTY:bool=on
  make -j 8
  make install

  # INSTALL OPENSLIDE
  cd $VIRTENV_ROOT/$VIRTENV

  wget https://github.com/openslide/openslide/releases/download/v3.4.1/openslide-3.4.1.tar.gz
  tar -xvf openslide-3.4.1.tar.gz

  cd $VIRTENV_ROOT/$VIRTENV/openslide-3.4.1
  CC=gcc CXX=g++ PKG_CONFIG_PATH=$VIRTENV_ROOT/$VIRTENV/lib/pkgconfig ./configure --prefix=$VIRTENV_ROOT/$VIRTENV
  make -j 8
  make install

  # LIBVIPS PART
  cd $VIRTENV_ROOT/$VIRTENV

  wget https://github.com/libvips/libvips/releases/download/v8.9.2/vips-8.9.2.tar.gz
  tar -xvf vips-8.9.2.tar.gz

  cd $VIRTENV_ROOT/$VIRTENV/vips-8.9.2
  CC=gcc CXX=g++ PKG_CONFIG_PATH=$VIRTENV_ROOT/$VIRTENV/lib/pkgconfig ./configure --prefix=$VIRTENV_ROOT/$VIRTENV
  make -j 8

  module purge
  module load pre2019
  module load cmake/2.8.11

  make install
fi

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
source $VIRTENV_ROOT/$VIRTENV/bin/activate

export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_NCCL_HOME=$EBROOTNCCL
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_BROADCAST=NCCL
export HOROVOD_WITHOUT_GLOO=1
export HOROVOD_WITH_TENSORFLOW=1

# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"

if [ ! -z $1 ] && [ $1 = 'create' ]; then
  pip install --upgrade pip --no-cache-dir
  pip install scikit-learn --no-cache-dir
  pip install Pillow --no-cache-dir
  pip install tqdm --no-cache-dir
  pip install six --no-cache-dir
  pip install opencv-python --no-cache-dir
  pip install openslide-python --no-cache-dir
  pip install pyvips --no-cache-dir
  pip install tensorflow==2.3.0 --no-cache-dir
  pip install horovod[tensorflow] --no-cache-dir
fi

cd $PROJECT_DIR