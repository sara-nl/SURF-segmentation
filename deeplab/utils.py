import tensorflow as tf
import horovod.tensorflow as hvd
import pdb
import psutil
import humanize
import numpy as np
import time
import GPUtil as GPU
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
import sys
import os
import pdb

sys.path.insert(0, os.path.join(os.getcwd(), 'keras-deeplab-v3-plus-master'))
from model import Deeplabv3
import numpy as np
import time

# tf.debugging.set_log_device_placement(True)



def printm():
    GPUs = GPU.getGPUs()
    print(f"Found {len(GPUs)}")
    process = psutil.Process(os.getpid())
    print(f"Gen RAM Free: {humanize.naturalsize(psutil.virtual_memory().available)}",
          " I Proc size: {humanize.naturalsize( process.memory_info().rss)}")
    for i, gpu in enumerate(GPUs):
        print(
            f"GPU:{i} RAM Free: {gpu.memoryFree}MB | Used: {gpu.memoryUsed}MB | Util {gpu.memoryUtil * 100}% | Total {gpu.memoryTotal}MB")


def mindevice():
    GPUs = GPU.getGPUs()
    # memfree = [(gpu.memoryUsed, gpu) for GPUs ]
    return 1


def init(opts):
    """ Run initialisation options"""

    if opts.horovod:
        hvd.init()

        if hvd.rank() == 0: print("Now hvd.init")
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        if opts.cuda:
            gpus = tf.config.experimental.list_physical_devices('GPU')

            print(gpus)
            if hvd.rank() == 0: print("hvd.size() = ", hvd.size())
            # print("GPU's", gpus, "with Local Rank", hvd.local_rank())
            # print("GPU's", gpus, "with Rank", hvd.rank())

            if gpus:
                print(f"pysical device setting: {gpus[hvd.local_rank() % 4]}")
                # tf.config.experimental.set_visible_devices(gpus[hvd.local_rank() % 4], 'GPU')
                # tf.config.experimental.set_memory_growth(gpus[hvd.local_rank() % 4], True)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if hvd.rank() == 0:
        print("Past hvd.init()")


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.

    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.

    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.

    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """
    if global_step > 0:
        global_step = global_step % total_steps
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


def cyclic_learning_rate(global_step, base_lr=0.001, max_lr=0.006, step_size=2000., gamma=1):

    def scale_fn(step):
        return gamma ** step

    cycle = np.floor(1 + global_step / (2 * step_size))
    x = np.abs(global_step / step_size - 2 * cycle + 1)

    return base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * scale_fn(global_step)


def get_model_and_optimizer(opts):
    """ Load the model and optimizer """

    if opts.evaluate:
        assert opts.model_dir, "WARNING: Please provide --model_dir when --evaluate"
    
    if opts.model_dir:
        print(f'Resuming model from {opts.model_dir}...')
        model = tf.keras.models.load_model(opts.model_dir)
    else:
        model = Deeplabv3(input_shape=(opts.image_size, opts.image_size, 3), classes=2, backbone='xception',opts=opts)
        

    if opts.horovod:
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if opts.fp16_allreduce else hvd.Compression.none

        if opts.optimizer == 'Adam':
            opt = tf.optimizers.Adam(opts.base_lr * hvd.size(), epsilon=opts.epsilon)
        elif opts.optimizer == 'SGD':
            opt = tf.optimizers.SGD(opts.base_lr * hvd.size(), opts.momentum, opts.nesterov)
        else:
            raise NotImplementedError('Only SGD and Adam are supported for now')

        # opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

        # Horovod: add Horovod DistributedOptimizer.
        # opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=5, op=hvd.Adasum)

    else:
        if opts.optimizer == 'Adam':
            opt = tf.optimizers.Adam(opts.base_lr, epsilon=opts.epsilon)
        elif opts.optimizer == 'SGD':
            opt = tf.optimizers.SGD(opts.base_lr, opts.momentum, opts.nesterov)
        else:
            raise NotImplementedError('Only SGD and Adam are supported for now')
        compression = None

    if hvd.rank() == 0:
        print("Compiling model...")

    model.layers[0].build(input_shape=(None, opts.image_size, opts.image_size, 3))
    # for layer in model.layers[0].layers:
    #     for var in layer.variables:
    #         print(var.name, var.shape, var.device)


    if hvd.rank() == 0:
        model.summary()
        # if opts.model == 'deeplab':
        #     for layer in model.layers: print(layer.name,layer.dtype)
        # else:
        #     for layer in model.layers[0].layers: print(layer.name,layer.dtype)

    return model, opt, compression


def filter_fn(image, mask):
    """ Filter images for images with tumor and non tumor """
    return tf.math.zero_fraction(mask) >= 0.2


def setup_logger(opts):
    """ Setup the tensorboard writer """
    # Sets up a timestamped log directory.

    logdir = f'{opts.log_dir}_{str(opts.image_size)}'
    if hvd.rank() == 0:
        if not os.path.exists(logdir):
            os.makedirs(logdir)

    if opts.horovod:
        # Creates a file writer for the log directory.
        if hvd.rank() == 0:
            file_writer = tf.summary.create_file_writer(logdir)
        else:
            file_writer = None
    else:
        # If running without horovod
        file_writer = tf.summary.create_file_writer(logdir)

    if opts.evaluate:
        if not os.path.exists(os.path.join(opts.log_dir, 'test_masks')):
            os.makedirs(os.path.join(opts.log_dir, 'test_masks'))

    return file_writer


def log_training_step(opts, model, file_writer, x, y, loss, pred, step, metrics, optimizer, steptime,epoch):
    """ Log to file writer during training"""
    if hvd.local_rank() == 0 and hvd.rank() == 0:

        # Make y [batch_size,image_size,image_size,1], prepare for metrics
        y = tf.argmax(y,axis=-1)[...,None]
        compute_loss, compute_miou, compute_auc = metrics
        
        compute_miou.update_state(y,pred)
        compute_auc.update_state(y,pred)

        # Training Prints
        tf.print('\nEpoch:',epoch,'Step', step, '/', opts.steps_per_epoch,
                 ': loss', loss,
                 ': miou', compute_miou.result().numpy(),
                 ': auc', compute_auc.result().numpy(), '\n')
        
        with file_writer.as_default():

            image = tf.cast(255 * x, tf.uint8)
            mask = tf.cast(255 * y, tf.uint8)
            summary_predictions = tf.cast(tf.expand_dims(pred * 255, axis=-1), tf.uint8)
            tf.summary.scalar('Training StepTime', steptime, step=tf.cast(step, tf.int64))
            tf.summary.image('Train_image', image, step=tf.cast(step, tf.int64), max_outputs=2)
            tf.summary.image('Train_mask', mask, step=tf.cast(step, tf.int64), max_outputs=2)
            tf.summary.image('Train_prediction', summary_predictions, step=tf.cast(step, tf.int64),
                             max_outputs=2)
            tf.summary.scalar('Training Loss', loss, step=tf.cast(step, tf.int64))
            tf.summary.scalar('Training mIoU', compute_miou.result().numpy(),step=tf.cast(step, tf.int64))
            tf.summary.scalar('Training AUC', compute_auc.result().numpy(), step=tf.cast(step, tf.int64))

            # Logging the optimizer's hyperparameters
            for key in optimizer._hyper:
                tf.summary.scalar(key, optimizer._hyper[key].numpy(), step=tf.cast(step, tf.int64))
            # Extract weights and filter out None elemens for aspp without weights
            weights = filter(None, [x.weights for x in model.layers])
            for var in weights:
                tf.summary.histogram('%s' % var[0].name, var[0], step=tf.cast(step, tf.int64))

        file_writer.flush()

    return


def log_validation_step(opts, file_writer, image, mask, step, pred, val_loss, val_miou, val_auc,epoch):
    """ Log to file writer after a validation step """
    if hvd.local_rank() == 0:
        with file_writer.as_default():
            tf.summary.image(f'Validation image of worker {hvd.rank()}', image, step=tf.cast(step, tf.int64),
                             max_outputs=5)
            tf.summary.image(f'Validation mask of worker {hvd.rank()}', mask, step=tf.cast(step, tf.int64),
                             max_outputs=5)
            tf.summary.image(f'Validation_prediction of worker {hvd.rank()}', pred, step=tf.cast(step, tf.int64),
                             max_outputs=2)
            tf.summary.scalar(f'Validation Loss of worker {hvd.rank()}', val_loss, step=tf.cast(step, tf.int64))
            tf.summary.scalar(f'Validation Mean IoU of worker {hvd.rank()}', val_miou.result().numpy(), step=tf.cast(step, tf.int64))
            tf.summary.scalar('Validation AUC', val_auc.result().numpy(), step=tf.cast(step, tf.int64))

        file_writer.flush()

        tf.print('Validation at epoch',epoch,'Step', step, '/', opts.steps_per_epoch,
                 ': validation loss', val_loss,
                 ': validation miou', val_miou.result().numpy(),
                 ': validation auc', val_auc.result().numpy())
    return
