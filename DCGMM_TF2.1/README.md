# WP3 - EXAMODE COLOR INFORMATION

Internal repository regarding WP3 of Examode project concerning the color information in Whole Slide Images

- This repository is enabling color transformation on the basis of :
    - < a href=https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization>Deep Convolutional Gaussian Mixture Model for Stain-Color Normalization in Histopathological H&E Images</a>

However it adds the following features:

- All is implemented in TF2.2.0
- Batch Size higher then 1 is possible in a multi-GPU setting
- Color Information measures can be evaluated using: 
    - Normalized Median Intensity (NMI) measure
    - Standard deviation of NMI
    - Coefficient of variation of NMI
    ref: < a href=https://pubmed.ncbi.nlm.nih.gov/26353368/>Stain Specific Standardization of Whole-Slide Histopathological Images</a>



# Setup
These steps ran on LISA this module environment, where we first clone and enable the 2020 software stack: 

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
module load Python/3.7.4-GCCcore-8.3.0
module load cuDNN/7.6.5.32-CUDA-10.1.243

```
Install requirements:
```
pip install -r requirements.txt
```


