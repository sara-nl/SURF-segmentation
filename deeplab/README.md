# Keras implementation of Deeplabv3+
DeepLab is a deep learning model for semantic image segmentation.  

Model is based on the original TF frozen graph. It is possible to load pretrained weights into this model. Weights are directly imported from original TF checkpoint.  


## How to train this model
### Model Training

- Configuration of model training and evaluation is done through `SURF-segmentation/deeplab/options.py`
> See python file for explanation of configuration.

### Distributed Training
- Multi worker training is facilitated through the <a href="https://horovod.readthedocs.io/en/stable/library">horovod</a> library
- Store the `hosts:number_of_slots` in the hosts variable 
- With <a href="https://slurm.schedmd.com/documentation.html">SLURM</a> this is done by:

```
#!/bin/bash
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

# Running on LISA
To start a training run on LISA with the **CAMELYON17** dataset, 

- **image size 1024x1024**

- **batch size 2**

- **4 workers**

training on medical center 1, and validating on medical center 2 (See https://camelyon17.grand-challenge.org/):

### Training:
```
horovodrun -np 8 \
--mpi-args="--map-by ppr:4:node" \
--hosts $hosts \
python -u train.py \
--image_size 1024 \
--batch_size 1 \
--verbose debug \
--fp16_allreduce \
--log_dir /home/rubenh/SURF-segmentation/deeplab/logs/ \
--log_every 1 \
--min_lr 0.0 \
--max_lr 0.001 \
--validate_every 60000 \
--steps_per_epoch 50000 \
--slide_path '/nfs/managed_datasets/CAMELYON17/training/center_1/' \
--label_path '/nfs/managed_datasets/CAMELYON17/training' \
--valid_slide_path '/nfs/managed_datasets/CAMELYON17/training/center_2/' \
--valid_label_path '/nfs/managed_datasets/CAMELYON17/training/' \
--bb_downsample 7 \
--batch_tumor_ratio 1 \
--optimizer Adam \
--lr_scheduler cyclic
```
## How to load model
- Provide `--model_dir` to options of model training.

## How to evaluate this model
- Set `--evaluate` to options of model training.

## Efficient Multi Worker Training
- Set:
```
horovodrun -np 8 \
--mpi-args="--map-by ppr:4:node" \
--hosts $hosts \
--autotune \
--autotune_log_file autotune.csv
```
## CAMELYON16 Checkpoint

- The following checkpoint(s) are available for DeepLabV3+:

| 	Dataset 	  | 	 Folder 	    |   Patch Size   | Train Time      |     Iteratios (# images )  |    Validation mIoU  |
| ----------------------- | ----------------------- | -------------- | --------------- | -------------------------- |-------------------- | 
|  CAMELYON16             | deeplab-camelyon16      | 1024           | 12:40:30.002    | 50000            	    |       0.8899        |                                
- Provide this checkpoint to `--model_dir` or load it directly using \href{https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model}[`tf.keras.models.load_model`]                         
