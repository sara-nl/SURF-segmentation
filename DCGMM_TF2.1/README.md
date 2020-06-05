# WP3 - EXAMODE COLOR INFORMATION

Internal repository regarding WP3 of Examode project concerning the color information in Whole Slide Images

- This repository is enabling color transformation on the basis of :
    - <a href="https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization">Deep Convolutional Gaussian Mixture Model for Stain-Color Normalization in Histopathological H&E Images</a>

However it adds the following features:

- All is implemented in TF2.2.0
- Batch Size higher then 1 is possible in a multi-GPU setting
- Color Information measures can be evaluated using: 
    - Normalized Median Intensity (NMI) measure
    - Standard deviation of NMI
    - Coefficient of variation of NMI
    
    ref: <a href="https://pubmed.ncbi.nlm.nih.gov/26353368/">Stain Specific Standardization of Whole-Slide Histopathological Images</a>


<p align="center">
<img  width="250" height="250" src=_images/template.png> ==> <img  width="250" height="250" src=_images/clusters.png>
</p>  
> The tissue class membership, computed by the DCGMM (right)

# Setup
These steps ran on LISA this module environment, where we first clone and enable the 2020 software stack: 

```
cd ~
git clone https://git.ia.surfsara.nl/environment-modules/environment-modules-lisa.git
```

Load Modules:
```
module purge
module use ~/environment-modules-lisa
module load 2020
module load Python/3.7.4-GCCcore-8.3.0
module load cuDNN/7.6.5.32-CUDA-10.1.243

```
Install requirements:
```
pip install -r requirements.txt
```

Options:


```
python main.py --help
>>>>>
TF2 DCGMM model

optional arguments:
  -h, --help            show this help message and exit
  --eval_mode           Run in evaluation mode. If false, training mode is
                        activated (default: False)
  --img_size IMG_SIZE   Image size to use (default: 256)
  --batch_size BATCH_SIZE
                        Batch size to use (default: 1)
  --epochs EPOCHS       Number of epochs for training. (default: 50)
  --num_clusters NUM_CLUSTERS
                        Number of tissue classes to use in DCGMM modelling
                        (default: 4)
  --dataset DATASET     Which dataset to use. "16" for CAMELYON16 or "17" for
                        CAMELYON17 (default: 16)
  --train_centers TRAIN_CENTERS [TRAIN_CENTERS ...]
                        Centers for training. Use -1 for all (default: [-1])
  --val_centers VAL_CENTERS [VAL_CENTERS ...]
                        Centers for validation. Use -1 for all (default: [-1])
  --train_path TRAIN_PATH
                        Folder of where the training data is located (default:
                        None)
  --valid_path VALID_PATH
                        Folder where the validation data is located (default:
                        None)
  --logdir LOGDIR       Folder where to log tensorboard and model checkpoints
                        (default: logs)
  --template_path TEMPLATE_PATH
                        Folder where template images are stored for
                        deployment. (default: template)
  --images_path IMAGES_PATH
                        Path where images to normalize are located (default:
                        images)
  --load_path LOAD_PATH
                        Path where to load model from (default:
                        logs/train_data)
  --save_path SAVE_PATH
                        Where to save normalized images (default: norm_images)
  --legacy_conversion   Legacy HSD conversion (default: True)
  --normalize_imgs      Normalize images between -1 and 1 (default: True)
  --log_every LOG_EVERY
                        Log every X steps during training (default: 100)
  --save_every SAVE_EVERY
                        Save a checkpoint every X steps (default: 5000)
  --debug               If running in debug mode (only 10 images) (default:
                        False)
  --val_split VAL_SPLIT
```
### Training
```
python3 main.py \
--img_size 256 \
--batch_size 16 \
--epochs 5 \
--num_clusters 4 \
--train_path /nfs/managed_datasets/CAMELYON17/training/center_1/patches_positive_256 \
--legacy_conversion
```

- This will train the DCGMM for 5 epochs, and save summaries and checkpoints in `/logs` (default)

### Evaluation
```
python3 main.py \
--img_size 256 \
--template_path /nfs/managed_datasets/CAMELYON17/training/center_1/patches_positive_256 \ # specify template directory
--images_path /nfs/managed_datasets/CAMELYON17/training/center_1/patches_positive_256 \   # specify target images directory
--load_path ~/examode/deeplab/DCGMM_TF2.1/logs/train_data/256-tr1-val1/checkpoint_4336 \  # specify checkpoint directory
--legacy_conversion \
--eval_mode \
--save_path saved_images                                                                  # specify save path to save transformed images
```

### TODO
- [ ] Implement multi node framework (Horovod)
