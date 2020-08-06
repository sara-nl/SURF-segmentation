# Extended FoV Semantic Segmentation
This repository is developed as part of the Examode EU project, and is meant for conducting experiments for large field-of-view semantic segmentation. The current codebase supports CAMELYON16 and CAMELYON17, and supports efficient execution on multi-node CPU clusters, as well as multi-node, multi-GPU clusters. Models using very large FoV (> 1024x1024) can be trained on multi-GPU cluster, using the instructions below. The models adapted for the use case of semantic segmentation of malignant tumor regions are:
- EfficientDet ( https://arxiv.org/abs/1911.09070 )
- DeeplabV3+ ( https://arxiv.org/abs/1802.02611 )

https://camelyon16.grand-challenge.org/

https://camelyon17.grand-challenge.org/

# Setup
These steps ran on LISA / Cartesius with this module environment, where we first clone and enable 2020: 


Modules loaded:
```
module purge
module load 2019
module load Python/3.6.6-foss-2019b

```
### Setting up the Python depencies (specific to LISA GPU)
Now export environment variables for installing Horovod w/ MPI for multiworker training, and install Python packages:
```
module purge
module load 2019
module load Python/3.6.6-foss-2019b
module load cuDNN/7.6.5.32-CUDA-10.1.243
source ~/virtualenvs/openslide/bin/activate
pip install tensorflow==2.3.0
pip install scikit-learn 
pip install Pillow 
pip install tqdm 
pip install six
pip install opencv-python
pip install pyvips
pip install openslide-python
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_NCCL_HOME=$EBROOTNCCL
export HOROVOD_GPU_ALLREDUCE=NCCL
#export HOROVOD_GPU_ALLGATHER=MPI
#export HOROVOD_GPU_BROADCAST=MPI
export PATH=/home/$USER/virtualenvs/openslide/bin:$PATH
export LD_LIBRARY_PATH=/home/$USER/virtualenvs/openslide/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/$USER/virtualenvs/openslide/lib:$LD_LIBRARY_PATH
export CPATH=/home/$USER/virtualenvs/openslide/include:$CPATH
# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"
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
Multi - GPU TensorFlow model

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
  --model {effdetd0,effdetd4,deeplab}
                        EfficientDet or Deeplabv3+ model for semantic
                        segmentation. (default: effdetd0)
  --no_cuda             Use CUDA or not (default: False)
  --horovod             Distributed training via horovod (default: True)
  --fp16_allreduce      Reduce to FP16 precision for gradient all reduce
                        (default: False)
  --train_centers TRAIN_CENTERS [TRAIN_CENTERS ...]
                        Centers for training. Use -1 for all, otherwise 2 3 4
                        eg. (default: [-1])
  --val_centers VAL_CENTERS [VAL_CENTERS ...]
                        Centers for validation. Use -1 for all, otherwise 2 3
                        4 eg. (default: [-1])
  --hard_mining         Use hard mining or not (default: False)
  --slide_path SLIDE_PATH
                        Folder of where the training data whole slide images
                        are located (default: None)
  --label_path LABEL_PATH
                        Folder of where the training data whole slide images
                        labels are located (default: None)
  --valid_slide_path VALID_SLIDE_PATH
                        Folder of where the validation data whole slide images
                        are located (default: None)
  --valid_label_path VALID_LABEL_PATH
                        Folder of where the validation data whole slide images
                        labels are located (default: None)
  --weights_path WEIGHTS_PATH
                        Folder where the pre - trained weights is located
                        (default: None)
  --slide_format SLIDE_FORMAT
                        In which format the whole slide images are saved.
                        (default: tif)
  --label_format LABEL_FORMAT
                        In which format the labels are saved. (default: tif)
  --valid_slide_format VALID_SLIDE_FORMAT
                        In which format the whole slide images are saved.
                        (default: tif)
  --valid_label_format VALID_LABEL_FORMAT
                        In which format the labels are saved. (default: tif)
  --data_sampler {radboud,surf}
                        Which DataSampler to use (default: radboud)
  --bb_downsample BB_DOWNSAMPLE
                        Level to use for the bounding box construction as
                        downsampling level of whole slide image (default: 7)
  --batch_tumor_ratio BATCH_TUMOR_RATIO
                        The ratio of the batch that contains tumor (default:
                        1)
  --sample_processes SAMPLE_PROCESSES
                        Amount of Python Processes to start for the Sampler
                        (default: 1)
  --resolution RESOLUTION
                        The resolution of the patch to extract (in micron per
                        pixel) (default: 0.25)
  --label_map KEY:VAL [KEY:VAL ...]
                        Add label_map for Radboud datasampler as dictionary 
                        argument like so label1:mapping1 label2:mapping2 
                        (default: None)
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
To start a training run on LISA with the CAMELYON16 dataset, image size 1024x1024 and batch size 2, on 4 workers:

- DataSampler from SURF

```
mpirun -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH python -u train.py \
--img_size 1024 \
--horovod \
--model effdetd0 \
--batch_size 2 \
--fp16_allreduce \
--log_dir /home/rubenh/examode/deeplab/CAMELYON_TRAINING/logs/test/ \
--log_every 2 \
--num_steps 5000 \
--slide_format tif \
--label_format tif \
--slide_path /nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor \
--label_path /nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/Mask \
--bb_downsample 7 \
--batch_tumor_ratio 0.5 \
--log_dir /home/rubenh/SURF-deeplab/TRAINING/logs/test/
--validate_every 4096
```
- DataSampler from Radboud
```
mpirun -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH python -u train.py \
--img_size 1024 \
--horovod \
--model effdetd0 \
--batch_size 1 \
--fp16_allreduce \
--log_dir /home/rubenh/SURF-deeplab/TRAINING/logs/test/ \
--log_every 2 \
--num_steps 100000 \
--slide_format tif \
--label_format xml \
--valid_slide_format tif \
--valid_label_format xml \
--slide_path /nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor \
--label_path /nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/XML \
--valid_slide_path /nfs/managed_datasets/CAMELYON16/Testset/Images \
--valid_label_path /nfs/managed_datasets/CAMELYON16/Testset/Ground_Truth/Annotations \
--data_sampler radboud \
--label_map _0:0 _2:1 \
--sample_processes 1 \
--validate_every 4096
```
See the *run_train.sh* file for details

A training run on positive patches of 1024 x 1024 will converge in 2 hours on 4 TITANRTX nodes (Batch Size 2, ppr:4:node) to mIoU ~ 0.90

# Running on multi-node CPU clusters
To start a training run on CPU-based cluster with the CAMELYON16 dataset, image size 2048x2048 and batch size 2 per worker, using 64 workers:

```
mpiexec.hydra -map-by ppr:1:socket -np 64 python -u train.py --no_cuda --horovod --img_size 2048 --dataset 16 --log_dir "logs" --fp16_allreduce --val_split 0.15 --weights_path 'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5' --log_every 512 --validate_every 4096
```

A training run on positive patches of 2048x2048 will converge in ~5 hours on 32 Cascade Lake 8260 CPU nodes (Batch Size 2 per worker, 2 workers per node) to mIoU ~ 0.97

## Research
If this repository has helped you in your research we would value to be acknowledged in your publication.

# Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825292. This project is better known as the ExaMode project. The objectives of the ExaMode project are:
1. Weakly-supervised knowledge discovery for exascale medical data.  
2. Develop extreme scale analytic tools for heterogeneous exascale multimodal and multimedia data.  
3. Healthcare & industry decision-making adoption of extreme-scale analysis and prediction tools.

For more information on the ExaMode project, please visit www.examode.eu. 

![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/horizon.jpg)  ![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/flag_yellow.png) <img src="https://www.examode.eu/wp-content/uploads/2018/11/cropped-ExaModeLogo_blacklines_TranspBackGround1.png" width="80">

