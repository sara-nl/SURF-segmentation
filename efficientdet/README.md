### TODO
- [x] Implement multi node framework (Horovod)
- [ ] Provide Checkpoints
- [ ] Give overview of different Horovod autotune settings
- [ ] Give explanation in README on testing Whole Slide Images in CAMELYON17


# Extended FoV Semantic Segmentation using EfficientDet
- EfficientDet ( https://arxiv.org/abs/1911.09070 )

- The methods described here are based on the _tf.keras_ implementation
of the EfficientDet network in the `keras/` folder. The Google AutoML implementation
also has a _tf.Estimator_ implementation. See https://github.com/google/automl/tree/master/efficientdet

https://camelyon16.grand-challenge.org/

https://camelyon17.grand-challenge.org/

# Setup
These steps ran on LISA with this module environment: 

- First install a virtual environment with OpenSlide and PyVips from https://github.com/sara-nl/SURF-deeplab.
- This will install the libraries needed for processing of Whole Slide Images.

Modules loaded:
```
VENV_NAME=openslide-pyvips
cd $HOME
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

```

## Dependencies
```
# Set Environment Variables
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
```

## Install requirements:
```
pip install -r SURF-segmentation/efficientdet/keras/requirements.txt
```


# Model Training

- Configuration of model training and evaluation is done through `SURF-segmentation/efficientdet/hparams_config.py`
> See python file for explanation of configuration
> **If there are no code comments above a configuration, then it is not used in the SURF EfficientDet**
- The configurations in this file may be overwritten by command line arguments in `SURF-segmentation/efficientdet/keras/run.sh`

## Distributed Training
- Multi worker training is facilitated through the <a href="https://horovod.readthedocs.io/en/stable/library">horovod</a> library
- Store the `hosts:number_of_slots` in the hosts variable 
- In <a href="https://slurm.schedmd.com/documentation.html">SLURM</a> this is done by:

```
hosts=""
number_of_slots=4
for host in $(scontrol show hostnames);
do
	hosts="$hosts$host:$number_of_slots,"
done
hosts="${hosts%?}"
echo "HOSTS: $hosts"
```
- This will output `r34n4:4` if running on node r34n4 with 4 GPU's


- Configurable command line arguments:
```
python -u segmentation.py --help
>>>>
optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size to use
  --steps_per_epoch STEPS_PER_EPOCH
                        Steps per epoch
  --num_epochs NUM_EPOCHS
                        Epochs
  --name {efficientdet-d0,efficientdet-d4}
                        EfficientNet backbone.
  --evaluate            If only perform evaluation
  --optimizer {Adam,SGD}
  --lr_decay_method {stepwise,cosine,polynomial,cyclic}
  --step_size STEP_SIZE
                        Step_size for the cyclic learning rate
  --gamma GAMMA         Decay parameter for the cyclic learning rate
  --log_dir LOG_DIR     Folder of where the logs are saved
  --run_name RUN_NAME
```

# Running on LISA
To start a training run on LISA with the **CAMELYON17** dataset, 

**image size 1024x1024**

**batch size 1**

**4 workers**

training on medical center 1, and validating on medical center 2 (See https://camelyon17.grand-challenge.org/):

### Training:
- Set in `SURF-segmentation/efficientdet/hparams_config.py`:
    
```
h.slide_path = '/nfs/managed_datasets/CAMELYON17/training/center_1/'

h.label_path = '/nfs/managed_datasets/CAMELYON17/training'

h.valid_slide_path = '/nfs/managed_datasets/CAMELYON17/training/center_2/'

h.valid_label_path = '/nfs/managed_datasets/CAMELYON17/training/'

h.image_size = 1024  

h.img_size = 1024

```
    

```
horovodrun -np 4 \
--autotune \
--autotune_log_file autotune.csv \
--mpi-args="--map-by ppr:4:node" \
--hosts $hosts \
python -u segmentation.py \
--batch_size 1 \
--optimizer SGD \
--lr_decay_method cosine \
--name efficientdet-d0 \
--log_dir /home/rubenh/SURF-deeplab/efficientdet/keras/test \
--steps_per_epoch 100 \
--num_epochs 500
```

- This will train the model for 500 epochs, while autotuning communication settings with <a href="https://horovod.readthedocs.io/en/stable/autotune_include.html">horovod autotune</a>, and saving these settings to autotune.csv.     


### Evaluation:
- This will evaluate the sampler type given to `evaluate()` in `SURF-segmentation/efficientdet/keras/segmentation.py`
- It will save csv files containing the metastases (negative, itc, micro, macro) per Whole Slide Image

```
horovodrun -np 4 \
--autotune \
--autotune_log_file autotune.csv \
--mpi-args="--map-by ppr:4:node" \
--hosts $hosts \
python -u segmentation.py \
--batch_size 1 \
--evaluate \
--optimizer SGD \
--lr_decay_method cosine \
--name efficientdet-d0 \
--log_dir /home/rubenh/SURF-deeplab/efficientdet/keras/test \
--steps_per_epoch 100 \
--num_epochs 500
