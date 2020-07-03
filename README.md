# DeepLabv3 for Exteded FoV Semantic Segmentation
This repository is developed as part of the Examode EU project, and is meant for conducting experiments for large field-of-view semantic segmentation. The current codebase supports CAMELYON16 and CAMELYON17, and supports efficient execution on multi-node CPU clusters, as well as multi-node, multi-GPU clusters. Models using very large FoV (> 1024x1024) can be trained on multi-GPU cluster, using the instructions below.

https://camelyon16.grand-challenge.org/

https://camelyon17.grand-challenge.org/

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
### Setting up the Python depencies (specific to LISA GPU)
First we will load the modules needed:
```
module purge
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243 

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
cd ~/SURF-deeplab/CAMELYON17_PREPROCESSING
# See flags in run_prepro.sh file for options
sh run_prepro.sh

```
- Options for model training:
```
mpirun -map-by ppr:1:node -np 1 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python -u train.py --help
>>>>
Multi - GPU TensorFlow DeeplabV3+ model

optional arguments:
  -h, --help            show this help message and exit
  --img_size IMG_SIZE   Image size to use (default: 1024)
  --batch_size BATCH_SIZE
                        Batch size to use (default: 2)
  --num_steps NUM_STEPS
                        Number of steps for training. A single step is defined
                        as one image. So a batch of 2 consists of 2 steps
                        (default: 50000)
  --val_split VAL_SPLIT
                        Part of images that is used as validation dataset,
                        validating on all images (default: 0.15)
  --no_cuda             Use CUDA or not (default: False)
  --horovod             Distributed training via horovod (default: True)
  --fp16_allreduce      Reduce to FP16 precision for gradient all reduce
                        (default: False)
  --dataset DATASET     Which dataset to use. "16" for CAMELYON16 or "17" for
                        CAMELYON17, this will load the CAMELYON datasets with
                        train / validation centers supplied for CAMELYON17
                        (default: 17)
  --train_centers TRAIN_CENTERS [TRAIN_CENTERS ...]
                        Centers for training. Use -1 for all, otherwise 2 3 4
                        eg. (default: [-1])
  --val_centers VAL_CENTERS [VAL_CENTERS ...]
                        Centers for validation. Use -1 for all, otherwise 2 3
                        4 eg. (default: [-1])
  --hard_mining         Use hard mining or not (default: False)
  --train_path TRAIN_PATH
                        Folder of where the training data is located (default:
                        None)
  --valid_path VALID_PATH
                        Folder where the validation data is located (default:
                        None)
  --weights_path WEIGHTS_PATH
                        Folder where the pre - trained weights is located
                        (default: None)
  --flip                Flip images for data augmentation (default: False)
  --random_crop         Randomly crop images for data augmentation (default:
                        False)
  --log_dir LOG_DIR     Folder of where the logs are saved (default: None)
  --log_every LOG_EVERY
                        Log every X steps during training (default: 128)
  --validate_every VALIDATE_EVERY
                        Run the validation dataset every X steps (default:
                        2048)
  --debug               If running in debug mode, only uses 100 images
                        (default: False)
  --pos_pixel_weight POS_PIXEL_WEIGHT
  --neg_pixel_weight NEG_PIXEL_WEIGHT
```

# Running on LISA
To start a training run on LISA with the CAMELYON16 dataset, image size 1024x1024 and batch size 2:
```
python train.py --dataset 16 --img_size 1024 --batch_size 2
```
OR (for multi-worker)
```
mpirun -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python train.py --dataset 16 --img_size 1024 --batch_size 2
```
See the *run_deeplab_exa.sh* file for details

A training run on positive patches of 1024 x 1024 will converge in 2 hours on 4 TITANRTX nodes (Batch Size 2, ppr:4:node) to mIoU ~ 0.90

# Running on multi-node CPU clusters
To start a training run on CPU-based cluster with the CAMELYON16 dataset, image size 2048x2048 and batch size 2 per worker, using 64 workers:

```
mpiexec.hydra -map-by ppr:1:socket -np 64 python -u train.py --no_cuda --horovod --img_size 2048 --dataset 16 --log_dir "logs" --fp16_allreduce --val_split 0.15 --weights_path 'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5' --log_every 512 --validate_every 4096
```

A training run on positive patches of 1024 x 1024 will converge in ~5 hours on 32 Cascade Lake 8260 CPU nodes (Batch Size 2 per worker, 2 workers per node) to mIoU ~ 0.97

## Research
If this repository has helped you in your research we would value to be acknowledged in your publication.

# Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825292. This project is better known as the ExaMode project. The objectives of the ExaMode project are:
1. Weakly-supervised knowledge discovery for exascale medical data.  
2. Develop extreme scale analytic tools for heterogeneous exascale multimodal and multimedia data.  
3. Healthcare & industry decision-making adoption of extreme-scale analysis and prediction tools.

For more information on the ExaMode project, please visit www.examode.eu. 

![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/horizon.jpg)  ![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/flag_yellow.png) <img src="https://www.examode.eu/wp-content/uploads/2018/11/cropped-ExaModeLogo_blacklines_TranspBackGround1.png" width="80">


