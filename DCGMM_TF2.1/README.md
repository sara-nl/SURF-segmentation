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



# Setup
These steps ran on LISA this module environment, where we first clone and enable the 2020 software stack: 

```
cd ~
git clone https://git.ia.surfsara.nl/environment-modules/environment-modules-lisa.git
module use ~/environment-modules-lisa
```

Load Modules:
```
module load 2020
module load Python/3.7.4-GCCcore-8.3.0
module load cuDNN/7.6.5.32-CUDA-10.1.243

```
Install requirements:
```
pip install -r requirements.txt
```

## Training
```
python3 main.py \
--img_size 256 \
--batch_size 16 \
--epochs 5 \
--num_clusters 4 \
--train_path /nfs/managed_datasets/CAMELYON17/training/center_1/patches_positive_256 \
--legacy_conversion
```

- This will train the DCGMM for 5 epochs, and save summaries and checkpoints in `/logs`

## Evaluation
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

