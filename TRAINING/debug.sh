clear

source init.sh

horovodrun -np 4 \
--mpi-args="--map-by ppr:4:node" \
--hosts localhost:4 \
python -u train.py \
--horovod \
--img_size 256 \
--model deeplab \
--batch_size 4 \
--fp16_allreduce \
--log_dir $PROJECT_DIR/logs/debug/ \
--log_every 2 \
--validate_every 5000 \
--num_steps 50000 \
--slide_path /nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor \
--label_path /nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/Mask \
--valid_slide_path /nfs/managed_datasets/CAMELYON16/Testset/Images \
--valid_label_path /nfs/managed_datasets/CAMELYON16/Testset/Ground_Truth/Masks \
--bb_downsample 7 \
--data_sampler surf \
--batch_tumor_ratio 1 \
--optimizer Adam \
--lr_scheduler cyclic
