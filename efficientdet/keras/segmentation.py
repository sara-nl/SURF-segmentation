# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A demo script to show to train a segmentation model."""
from absl import app
from absl import logging
import tensorflow as tf
import sys
sys.path.insert(0,'/home/rubenh/SURF-segmentation/TRAINING')
sys.path.insert(0,'/home/rubenh/SURF-segmentation/efficientdet')
from data_utils import SurfSampler, PreProcess
import numpy as np
import hparams_config
from keras import efficientdet_keras
import pdb
import time
from tqdm import tqdm
from glob import glob
import cv2
import horovod.tensorflow.keras as hvd
import train_lib
import os
import csv as csvlib
import pandas as pd
from openslide import OpenSlide, ImageSlide, OpenSlideUnsupportedFormatError
import difflib
from scipy import ndimage
import json
from sklearn.cluster import DBSCAN
from effdet_options import get_options
import util_keras

hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

print(gpus)
tf.debugging.set_log_device_placement(False)



def get_metastase(wsi_name,data,test_sampler,config):
    """
    
    
    Parameters
    ----------
    wsi_name : string ("patient_1_node_001")
        name of whole slide image.
    data : numpy.array
        data with mask predictions.
    config : config.object
        configuration file ("config.object")

    Computes the pN staging of lymph node of patients
    Negative:           smaller than 0.2 mm = 200 microm
    Micro-metastases:   200microm < ... < 2000 microm
    Macro-metastases:   > 2000 microm ( > 2 mm)
    
    For PN - Staging:
        See make_csv.py
    
    See https://camelyon17.grand-challenge.org/Evaluation/
    
    Returns
    -------
    int
        -.

    """

    csv = []
    try:
        # Load openslide
        if test_sampler.mode == 'validation':
            path = config.valid_slide_path
        elif test_sampler.mode == 'test':
            path = config.test_path
            
        slide_path = os.path.join(path,wsi_name+f'.{config.slide_format}')
        slide = OpenSlide(slide_path)
        # mask = OpenSlide(mask_path)
        
        # Get dimensions of level 0
        level_dimensions = (slide.level_dimensions[0][1],slide.level_dimensions[0][0])
        
        # Get resolution in microns per pixel (1 micrometer = 10e-6 meter = 0.001 mm)
        try:
            # CAM16
            resolution = (float(slide.properties['openslide.mpp-x']) + float(slide.properties['openslide.mpp-y'])) / 2
        except:
            # CAM 17 
            resolution = ( float(slide.properties['tiff.YResolution']) / float(slide.properties['openslide.level[0].height']) +
                           float(slide.properties['tiff.XResolution'])  / float(slide.properties['openslide.level[0].width']) ) / 2
        
        # Get resolution for micro metastases in micron per pixel
        res_for_micro = 200*resolution
        # Make mask numpy
        pred = np.zeros(level_dimensions,dtype=np.uint8)[...,None]

        # Fill mask with predictions
        print(f" Worker {hvd.rank()}: Starting filling mask of {slide_path}...")
        for idx, patch in enumerate(data):
            row = patch[0]
            column = patch[1]
            try:
                # Fill in based on coordinates
                pred[row:row+config.image_size,column:column+config.image_size] = patch[-1][0] # (1,config.image_size,config.image_size,1) -> (config.image_size,config.image_size,1))
                if idx % 100 == 0: print(f" Worker {hvd.rank()}:Processed patch {idx + 1} / {len(data)}...")
            except:
                print(f" Worker {hvd.rank()}:Skipping row {row} column {column} of patch {idx} (mask {pred.shape})")
                continue
    
        # Get minimum dimension so that resolution of 1 pixel is enough for micro metastases    
        micro_res_dimensions = (int(level_dimensions[1] // res_for_micro), int(level_dimensions[0] // res_for_micro))
        
        # Downsample predicted mask to this minimum dimension (if one positive pixel in this image, it is micrometastase)
        pred_down = cv2.resize(pred[0],dsize=(micro_res_dimensions))
        
        # Convolute downsampled image, so that if more then 10 grouped positive pixels, in sliding window sum(255*10 pixels) will be macro metastase
        _filter = np.ones((10,10))
        filtered = ndimage.convolve(pred_down, _filter, mode='constant', cval=0.0)
        
        if filtered.max() >= 2550:
            csv.append((slide_path.split('/')[-1],'macro'))
        elif pred_down.max():
            csv.append((slide_path.split('/')[-1],'micro'))   
        elif pred.max():
            csv.append((slide_path.split('/')[-1],'itc'))   
        else:
            csv.append((slide_path.split('/')[-1],'negative'))
    except Exception as e:
        print(f' E: {e} slide {slide_path}')
        pass
         
    
    print(f" Worker {hvd.rank()}: Saving csv of {slide_path}...")
    csv = sorted(csv)

    with open(os.path.join(config.log_dir,f'{wsi_name}.csv'),'w') as out:
        csv_out=csvlib.writer(out)
        csv_out.writerow(['patient','stage'])
        for row in csv:
            csv_out.writerow(row)
            
    return True



def evaluate(model,config,test_sampler):
    done=0
    wsi_idx=0
    _array=[]
    save_mask = np.zeros((config.image_size,config.image_size,3))
    patch_idx = 0
    
    while not done:
        # Get test batch (in orderly fashion; past WSI's / ROI's / FOV coordinates are dropped)
        patches, masks = test_sampler.__getitem__(wsi_idx)
        # Get patch, mask from test_sampler
        for (patch,mask) in zip(patches,masks):
            patch = patch[None,...]
            mask = mask[None,...]
            # Predict patch
            pred = model(patch)[0]
            pred = tf.expand_dims(tf.argmax(pred,axis=-1), axis=-1)
            predictions = tf.cast(pred * 255, tf.uint8).numpy()
            # Make mask
            if test_sampler.mode == 'validation':
                mask = tf.cast(255 * mask, tf.uint8).numpy()
            
            # row    ,  columnn
            y_topleft,x_topleft = test_sampler.save_data[patch_idx]['coords']#[0]
            
            # Append predicted data to save array
            _array.append([y_topleft,x_topleft,mask.astype('uint8'),predictions.astype('uint8')])
            
            # Make predictions fit on downsampled rgb-image
            x,y = x_topleft // 2**config.bb_downsample, y_topleft // 2**config.bb_downsample
            dsize = config.image_size // 2**config.bb_downsample
            mask_down = cv2.resize(predictions[0,...],dsize=(dsize,dsize))[...,None]
            
            # Resize a mask to sizes of downsampled rgb_image
            save_mask = cv2.resize(save_mask,(test_sampler.save_data[patch_idx]['image'].shape[1],test_sampler.save_data[patch_idx]['image'].shape[0]))
            
            try:
                # Fill downsampled image with downsampled predictions
                save_mask[y:y+len(mask_down),x:x+len(mask_down),:]=mask_down
            except:
                continue
            
        # If a WSI is completed save: 
        # 1. *.npy files with [x_topleft,y_topleft,mask,predictions]
        # 2. Test masks (downsampled)
        # 3. Test image (downsampled, with tumor overlay)   
        if wsi_idx != test_sampler.wsi_idx:
            data = np.array(_array)
            # wsi_name = test_sampler.save_data[patch_idx-1]['file_name'].split('/')[-1][:-4]
            wsi_name = test_sampler.save_data[0]['file_name'].split('/')[-1][:-4]

            ### Get PN - Staging
            get_metastase(wsi_name,data,test_sampler,config)
            
            # Mark tumor in black, else green
            save_mask = np.where(save_mask,[0,0,0],[0,255,0]).astype('uint8')
            # Overlay
            save_mask = cv2.addWeighted(test_sampler.save_data[0]['image'],0.8,save_mask,0.2,1)
            cv2.imwrite(os.path.join(config.log_dir,wsi_name+'_mask.png'),save_mask)
            
            if test_sampler.mode == 'validation':
                print(f" Worker {hvd.rank()}: Computing mIoU...")
                mask = np.empty((1,config.image_size,config.image_size,1))
                pred  = np.empty((1,config.image_size,config.image_size,1))
                mIoU = tf.keras.metrics.MeanIoU(num_classes=2)
                
                for x in tqdm(_array): 
                    mask  = x[2]
                    pred  = x[3]
                    _sum = np.sum(pred)
                    if np.isnan(_sum):
                        print(f" Worker {hvd.rank()}: Skipping patch due to NaN in prediction")
                        pass
                    mask = np.where(mask,1,0)
                    pred = np.where(pred,1,0)
                    # Update confusion matrix of mIoU see https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU#update_state
                    mIoU.update_state(mask,pred)

                print(f" Worker {hvd.rank()}: mIoU of {wsi_name} is {mIoU.result().numpy()}")
                print(f" Worker {hvd.rank()}: Evaluated {test_sampler.wsi_idx} / {len(test_sampler.valid_paths)} test WSI's.")
            else:
                print(f" Worker {hvd.rank()}: Evaluated {test_sampler.wsi_idx} / {len(test_sampler.test_paths)} test WSI's.")
               
           
            wsi_idx = test_sampler.wsi_idx
            test_sampler.save_data = []
            patch_idx = 0
            # Increment done at last wsi
            if test_sampler.mode == 'test':
                if not (len(test_sampler.test_paths_new) - 1): 
                    done += 1 
                    wsi_idx = 0
            else: 
                if not (len(test_sampler.valid_paths_new) - 1): 
                    done += 1
                    wsi_idx = 0
        
    try:
        mIoU.reset_states()
    except:
        pass
    return 
    



def main(config):

    assert isinstance(config.image_size,int),"WARNING: Please make sure that the config.image_size is an integer"
    train_sampler = SurfSampler(config,mode='train')
    valid_sampler = SurfSampler(config,mode='validation')
    valid_data    = valid_sampler.__getitem__(0)    
    test_sampler  = SurfSampler(config,mode='test')
    
    
    optimizer, learning_rate = train_lib.get_optimizer(config)
    compression = hvd.Compression.fp16 if config.fp16_allreduce else hvd.Compression.none
    # Horovod: adjust learning rate based on number of GPUs.
    # scaled_lr = 0.001 * hvd.size()
    # optimizer = tf.optimizers.Adam(scaled_lr)
    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(optimizer,compression=compression)
        
    model = efficientdet_keras.EfficientDetNet(config=config)
    if os.path.isfile(os.path.join(config.log_dir,'checkpoint')):
        print(f"Loading checkpoint from {os.path.join(config.log_dir,'checkpoint')} ...")
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'],
                      experimental_run_tf_function=False)
        model.build((config.batch_size, config.image_size, config.image_size, 3))
        ckpt_path = tf.train.latest_checkpoint(config.log_dir)
        util_keras.restore_ckpt(model, ckpt_path, config.moving_average_decay)
    else:
        model.build((config.batch_size, config.image_size, config.image_size, 3))
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'],
                      experimental_run_tf_function=False,
                      run_eagerly=True)
   
        model.summary()
    
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    
        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),
    
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, initial_lr=scaled_lr, verbose=1),
        
        
        hvd.callbacks.LearningRateScheduleCallback(learning_rate,
                                                   start_epoch=0,
                                                   end_epoch=config.num_epochs,
                                                   staircase=True,
                                                   momentum_correction=True,
                                                   steps_per_epoch=config.steps_per_epoch // 2,
                                                   initial_lr=1)
    ]
    # Horovod: write logs on worker 0.
    verbose = 1 if hvd.rank() == 0 else 0
    if not config.evaluate:
        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        # if hvd.rank() == 0:
        cb_options = train_lib.get_callbacks(config, train_sampler, valid_sampler, profile=False)
        callbacks.extend(cb_options)
        model.fit(
            train_sampler,
            epochs=config.num_epochs,
            steps_per_epoch=config.steps_per_epoch,
            validation_data=valid_data,
            callbacks=callbacks,
            use_multiprocessing=False,
            validation_freq=1,
            verbose=verbose)
            
        print(f'Finished training\n')
            
        print("Starting Evaluation...")
        
    evaluate(model,config,valid_sampler)
    # if hvd.rank() == 0:
    #     print(f'Finished evaluation with exceptions:\n {test_sampler.exceptions}')
        
    # get_pn_staging(config,test_sampler)

if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)

  config = hparams_config.get_efficientdet_config('efficientdet-d0')
  opts = get_options()

  config.update(opts.__dict__)

  main(config)
