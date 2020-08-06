from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd())
import tensorflow as tf
import horovod.tensorflow as hvd
# import byteps.tensorflow as hvd
from pprint import pprint
import pdb

from options import get_options
from utils import init, get_model_and_optimizer, setup_logger, log_training_step, log_validation_step
from data_utils import SurfSampler, RadSampler
import time
from tqdm import tqdm



def rank00():
    if hvd.rank() == 0 and hvd.local_rank() == 0:
        return True
    
    
def train_one_step(model, opt, x, y, step, loss_func, compression,file_writer):
    
    
    with tf.GradientTape() as tape:
        logits = model(x,training=True)
        loss = loss_func(y, logits)
        
    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape, compression=compression) #  device_sparse='/cpu:0', device_dense='/cpu:0',
    grads = tape.gradient(loss, model.trainable_variables)

    opt.apply_gradients(zip(grads, model.trainable_variables))

    if step == -1:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

    pred = tf.argmax(logits, axis=-1)

    return loss, pred


def validate(opts, model, step, val_dataset, file_writer, metrics):
    """ Perform validation on the entire val_dataset """
    if hvd.local_rank() == 0 and hvd.rank() == 0:
        print(f"Starting Validation...")
        compute_loss, compute_miou, compute_auc = metrics
        val_loss, val_miou, val_auc = [], [], []

        for image, label in tqdm(val_dataset):
            val_pred_logits = model(image,training=False)
            val_pred = tf.math.argmax(val_pred_logits, axis=-1)
            val_loss.append(compute_loss(label, val_pred_logits))
            val_miou.append(compute_miou(label, val_pred))
            val_auc.append(compute_auc(label[:, :, :, 0], val_pred))

        val_loss = sum(val_loss) / len(val_loss)
        val_miou = sum(val_miou) / len(val_miou)
        val_auc = sum(val_auc) / len(val_auc)

        image = tf.cast(255 * image, tf.uint8)
        mask = tf.cast(255 * label, tf.uint8)
        summary_predictions = tf.cast(val_pred * 255, tf.uint8)

        if len(summary_predictions.shape) == 3 and summary_predictions.shape[-1] != 1:
            summary_predictions = summary_predictions[:, :, :, None]

        log_validation_step(opts,file_writer, image, mask, step, summary_predictions, val_loss, val_miou, val_auc)

        compute_miou.reset_states()
        compute_auc.reset_states()

    return


def train(opts, model, optimizer, file_writer, compression, sampler):

    
    step = 0

    # Define metrics
    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    compute_miou = tf.keras.metrics.MeanIoU(num_classes=2)
    compute_auc = tf.keras.metrics.AUC()
    metrics = (compute_loss, compute_miou, compute_auc)

    while step < opts.num_steps:
        
         # with tf.profiler.experimental.Profile('logs'):
            train_ds =  sampler.get_next(train=True)
            for patch, mask in train_ds:
    
                t1 = time.time()
                loss, pred = train_one_step(model, optimizer, patch, mask, step, compute_loss, compression,file_writer)
                if rank00(): print(f'Training step in {time.time() - t1} seconds')
                
                if step % opts.log_every == 0 and step > 0:
                    log_training_step(opts, model, file_writer, patch, mask, loss, pred, step, metrics)
        
                step += opts.batch_size * hvd.size()
                
                
                if step % opts.validate_every == 0:
                    # Only one sample for validation
                    valid_ds =  sampler.get_next(train=False)
                    for patch, mask in valid_ds:
                        validate(opts, model, step, valid_ds, file_writer, metrics)
        
                if step > opts.num_steps:
                    break
                    
        
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

    validate(opts, model, step, valid_ds, file_writer, metrics)
    # if hvd.local_rank() == 0 and hvd.rank() == 0:
    #     model.save('saved_model.h5')

if __name__ == '__main__':
    opts = get_options()
    pprint(vars(opts))
    # Run horovod init
    init(opts)
    file_writer = setup_logger(opts)
    if opts.data_sampler == 'surf':
        sampler = SurfSampler(opts)
    elif opts.data_sampler == 'radboud':
        sampler = RadSampler(opts)
    
    model, optimizer, compression = get_model_and_optimizer(opts)

    if rank00(): print('Preparing training...')
    train(opts, model, optimizer, file_writer, compression, sampler)
    if rank00(): print('Training is done')
