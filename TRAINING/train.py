from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.insert(0, os.getcwd())
import horovod.tensorflow as hvd
import tensorflow as tf
import pdb
import time
from PIL import Image
tf.debugging.set_log_device_placement(False)
# import byteps.tensorflow as hvd
from pprint import pprint
import numpy as np
import cv2
from options import get_options
from utils import init, setup_logger, log_training_step, log_validation_step,cosine_decay_with_warmup, get_model_and_optimizer
from data_utils import SurfSampler, RadSampler, PreProcess
import time
from tqdm import tqdm
import random

def rank00():
    if hvd.rank() == 0 and hvd.local_rank() == 0:
        return True


def start(sampler,opts):
    if sampler == 'surf':
        train_sampler = SurfSampler(opts)
        valid_sampler = SurfSampler(opts,training=False)
        test_sampler = SurfSampler(opts,training=False,evaluate=opts.evaluate)
    elif sampler == 'radboud':
        sampler = RadSampler(opts)
        
    preprocess = PreProcess(opts)
    
    return train_sampler, valid_sampler, preprocess, test_sampler
    
def train_one_step(model, opt, x, y, step, loss_func, compression,file_writer):
    
    with tf.GradientTape() as tape:
        logits = model(x,training=True)
        loss = loss_func(y, logits)
        # scaled_loss = opt.get_scaled_loss(loss)
    
    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape,compression=compression,op=hvd.Adasum)#,device_sparse='/gpu:2', device_dense='/gpu:2')
    # scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    # grads = opt.get_unscaled_gradients(scaled_gradients)
    if not opts.evaluate:
        lr = cosine_decay_with_warmup(global_step=step,
                                          learning_rate_base=0.001,
                                          warmup_learning_rate=0.00001,
                                          total_steps=opts.num_steps // 1,
                                          warmup_steps=2*hvd.size())
        opt = tf.keras.optimizers.SGD(learning_rate=lr*hvd.size(), momentum=0.9, nesterov=True)
        grads = tape.gradient(loss, model.trainable_variables)
    
        opt.apply_gradients(zip(grads, model.trainable_variables))
        if step == 0:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)

    pred = tf.argmax(logits, axis=-1)

    return loss, pred, opt





def validate(opts, model, step, val_dataset, file_writer, metrics):
    """ Perform validation on the entire val_dataset """
    if hvd.local_rank() == 0 and hvd.rank() == 0:
        print(f"Starting Validation...")
        compute_loss, compute_miou, compute_auc = metrics
        val_loss, val_miou, val_auc = [], [], []

        for image, label in val_dataset:
            val_pred_logits = model(image,training=False)
            val_pred = tf.math.argmax(val_pred_logits, axis=-1)
            val_loss.append(compute_loss(label, val_pred_logits))
            val_miou.append(compute_miou(label, val_pred))
            # val_auc.append(compute_auc(label[:, :, :, 0], val_pred))

        val_loss = sum(val_loss) / len(val_loss)
        val_miou = sum(val_miou) / len(val_miou)
        # val_auc = sum(val_auc) / len(val_auc)

        image = tf.cast(255 * image, tf.uint8)
        mask = tf.cast(255 * label, tf.uint8)
        summary_predictions = tf.cast(tf.expand_dims(val_pred * 255, axis=-1), tf.uint8)


        # if len(summary_predictions.shape) == 3 and summary_predictions.shape[-1] != 1:
        #     summary_predictions = summary_predictions[:, :, :, None]

        log_validation_step(opts,file_writer, image, mask, step, summary_predictions, val_loss, val_miou)#, val_auc)

        compute_miou.reset_states()
        # compute_auc.reset_states()

    return


def train(opts, model, optimizer, file_writer, compression,train_sampler,valid_sampler,preprocess):

    
    step = 0

    # Define metrics
    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    compute_miou = tf.keras.metrics.MeanIoU(num_classes=2)
    compute_auc = tf.keras.metrics.AUC()
    metrics = (compute_loss, compute_miou, compute_auc)

    # tf.profiler.experimental.start(opts.log_dir, tf.profiler.experimental.ProfilerOptions(host_tracer_level=3, python_tracer_level=0))
    # tf.profiler.experimental.start(opts.log_dir)
    ### 10 steps for measuring profile ###
    # opts.num_steps=32
    idx = train_sampler.train_paths.index(random.choice(train_sampler.train_paths)) 
    for step in range(0,opts.num_steps,hvd.size()*opts.batch_size):
        # with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            patches,masks =  train_sampler.__getitem__(idx)
            train_ds = preprocess.tfdataset(patches,masks)
            for patch, mask in train_ds:
                t1 = time.time()
                loss, pred, optimizer = train_one_step(model, optimizer, patch, mask, step, compute_loss, compression,file_writer)
                steptime = time.time() - t1
                if rank00(): print(f'\nTraining step in {steptime} seconds\n')
                
                if step % opts.log_every == 0 and step > 0:
                    log_training_step(opts, model, file_writer, patch, mask, loss, pred, step, metrics,optimizer,steptime)
        
                
                if step % opts.validate_every == 0 and step > 0:
                    # Only one sample for validation
                    patches,masks,save_data =  valid_sampler.__getitem__(idx)
                    valid_ds = preprocess.tfdataset(patches,masks)
                    
                    for patch, mask in valid_ds:
                        validate(opts, model, step, valid_ds, file_writer, metrics)
                    if rank00(): 
                        print(f'\nSaving model...\n')
                        model.save(os.path.join(opts.log_dir,'saved_model'),save_format="tf")
                                   
                if opts.hard_mining:         
                    # Bit ugly to define the function here, but it works
                    def filter_hard_mining(image, mask):
                        pred_logits = model(image)
                        pred = tf.math.argmax(pred_logits, axis=-1)
                        miou = tf.keras.metrics.MeanIoU(num_classes=2)(mask, pred)
                        # Only select training samples with miou less then 0.95
                        return  miou < 0.95
                    _len = tf.data.experimental.cardinality(train_ds)
                    train_ds = train_ds.filter(filter_hard_mining)
                    print(f"Hard mining removed {_len - tf.data.experimental.cardinality(train_ds)} images")
            for metric in metrics: 
                if str(type(metric)).find('metrics') > -1: metric.reset_states() 

    # tf.profiler.experimental.stop()
    # sys.exit(0)
    # validate(opts, model, step, valid_ds, file_writer, metrics)
    # if hvd.local_rank() == 0 and hvd.rank() == 0:
    #     model.save('saved_model.h5')


def test(opts, model, optimizer, file_writer, compression, sampler,preprocess):

    step=0
    # Define metrics
    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    compute_miou = tf.keras.metrics.MeanIoU(num_classes=2)
    compute_auc = tf.keras.metrics.AUC()
    metrics = (compute_loss, compute_miou, compute_auc)

    
    done=0
    wsi_idx=0
    _array=[]
    save_mask = np.zeros((opts.img_size,opts.img_size,3))
    while not done == opts.test_cycles:

        # Get test batch (in orderly fashion; past WSI's / ROI's / FOV coordinates are dropped)
        patches, masks = sampler.__getitem__(wsi_idx,train=False)
        
        test_ds = preprocess.tfdataset(patches,masks)
        for idx, (patch, mask) in test_ds.enumerate(start=0):
            t1 = time.time()
            idx = idx.numpy()
            loss, pred, optimizer = train_one_step(model, optimizer, patch, mask, step, compute_loss, compression,file_writer)
            steptime = time.time() - t1
            if rank00(): print(f'\nTest step in {steptime} seconds\n')
            
            mask = tf.cast(255 * mask, tf.uint8).numpy()
            predictions = tf.cast(pred * 255, tf.uint8).numpy()
            # x_topleft,y_topleft = past_coords[-opts.img_size**2]
            x_topleft,y_topleft = sampler.save_data[idx]['coords'][0]
            _array.append([x_topleft,y_topleft,mask.astype('uint8'),predictions[...,None].astype('uint8')])
            
            x,y = x_topleft // 2**opts.bb_downsample, y_topleft // 2**opts.bb_downsample
            dsize = opts.img_size // 2**opts.bb_downsample
            mask_down = cv2.resize(predictions[0,...],dsize=(dsize,dsize))[...,None]
            save_mask = cv2.resize(save_mask,(sampler.save_data[idx]['image'].shape[1],sampler.save_data[idx]['image'].shape[0]))
            save_mask[x:x+len(mask_down),y:y+len(mask_down),:]=mask_down
        # If a WSI is completed save: 
        # 1. *.npy files with [x_topleft,y_topleft,mask,predictions]
        # 2. Test masks (downsampled)
        # 3. Test image (downsampled, with tumor overlay)   
        if wsi_idx != sampler.wsi_idx:
            save_array = np.array(_array)
            wsi_name = sampler.save_data[idx]['file_name'].split('/')[-1][:-4]
            np.save(os.path.join(opts.log_dir,'test_masks',wsi_name),save_array)
            # Mark tumor in black, else green
            save_mask = np.where(save_mask,[0,0,0],[0,255,0]).astype('uint8')
            # Overlay
            save_mask = cv2.addWeighted(sampler.save_data[idx]['image'],0.8,save_mask,0.2,1)
            cv2.imwrite(os.path.join(opts.log_dir,wsi_name+'_mask.png'),save_mask)
            
            print(f"Computing mIoU...")
            mask = np.empty((1,opts.img_size,opts.img_size,1))
            pred  = np.empty((1,opts.img_size,opts.img_size,1))
            mIoU = tf.keras.metrics.MeanIoU(num_classes=2)
            for x in tqdm(_array): 
                mask  = x[2]
                pred  = x[3]
                _sum = np.sum(pred)
                if np.isnan(_sum):
                    print(f"Skipping patch due to NaN in prediction")
                    pass
                mask = np.where(mask,1,0)
                pred = np.where(pred,1,0)
                if mask.max():
                    mask = np.where(mask,1,0)
                    pred = np.where(pred,1,0)
                    mIoU.update_state(mask,pred)
                else:
                    mask = np.where(mask,0,1)
                    pred = np.where(pred,0,1)
                    mIoU.update_state(mask,pred)
            print(f"mIoU of {wsi_name} in {opts.test_cycles} test_cycle(s) is {mIoU.result().numpy()}")
            mIoU.reset_states()
            wsi_idx = sampler.wsi_idx
            sampler.save_data = []
            # Increment done at last wsi
            if not (len(sampler.valid_paths) - 1): 
                done += 1 
                        
                
                

if __name__ == '__main__':
    opts = get_options()
    pprint(vars(opts))
    # Run horovod init
    init(opts)
    file_writer = setup_logger(opts)
    train_sampler, valid_sampler, preprocess,test_sampler = start(opts.data_sampler,opts)

    model, optimizer, compression = get_model_and_optimizer(opts)
    
    if opts.evaluate:
        assert opts.model_dir, "ValueError: No model_dir given for evaluation (--model_dir <type=str>)"
        if rank00(): print('Preparing evaluation...')
        test(opts, model, optimizer, file_writer, compression, test_sampler,preprocess)
        if rank00(): print('Evaluation is done')
    else:
        if rank00(): print('Preparing training...')
        train(opts, model, optimizer, file_writer, compression,train_sampler,valid_sampler,preprocess)
        if rank00(): print('Training is done')
