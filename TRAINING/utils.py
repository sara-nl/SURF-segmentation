import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import sys
import os
import pdb
sys.path.insert(0, os.path.join(os.getcwd(), 'keras-deeplab-v3-plus-master'))
sys.path.insert(0, os.path.join(os.getcwd(), 'keras-efficientdet'))
from model import Deeplabv3
from efficientdet_keras import EfficientDetNet

def rank00():
    if hvd.rank() == 0 and hvd.local_rank() == 0:
        return True

def init(opts):
    """ Run initialisation options"""
    
    if opts.horovod:
        hvd.init()
        if rank00(): print("Now hvd.init")
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        if opts.cuda:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if rank00(): print("hvd.size() = ", hvd.size())
            print("GPU's", gpus, "with Local Rank", hvd.local_rank())
            print("GPU's", gpus, "with Rank", hvd.rank())

            # if gpus:
            #     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank() % 4], 'GPU')
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if rank00(): print("Past hvd.init()")


def get_model_and_optimizer(opts):
    """ Load the model and optimizer """
    if opts.model =='effdetd0':
        model = EfficientDetNet('efficientdet-d0')
    elif opts.model == 'effdetd4':
        model = EfficientDetNet('efficientdet-d4')
    elif opts.model == 'deeplab':
        model = Deeplabv3(input_shape=(opts.img_size, opts.img_size, 3), classes=2, backbone='xception',opts=opts)

    if opts.horovod:
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if opts.fp16_allreduce else hvd.Compression.none

        opt = tf.optimizers.Adam(0.001 * hvd.local_size(), epsilon=1e-7)
        opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

        # Horovod: add Horovod DistributedOptimizer.
        # opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=5, op=hvd.Adasum)

    else:
        opt = tf.optimizers.Adam(0.0001, epsilon=1e-1)

    if rank00(): print("Compiling model...")

    model.build(input_shape=(None,opts.img_size, opts.img_size, 3))

    if rank00(): model.summary()

    return model, opt, compression


def filter_fn(image, mask):
    """ Filter images for images with tumor and non tumor """
    return tf.math.zero_fraction(mask) >= 0.2


def setup_logger(opts):
    """ Setup the tensorboard writer """
    # Sets up a timestamped log directory.
    
    logdir = f'{opts.log_dir}_{str(opts.img_size)}' 
    if rank00():
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    
    if opts.horovod:
        # Creates a file writer for the log directory.
        if rank00():
            file_writer = tf.summary.create_file_writer(logdir)
        else:
            file_writer = None
    else:
        # If running without horovod
        file_writer = tf.summary.create_file_writer(logdir)

    return file_writer


def log_training_step(opts, model, file_writer, x, y, loss, pred, step, metrics,optimizer):
    """ Log to file writer during training"""
    if hvd.local_rank() == 0 and hvd.rank() == 0:

        compute_loss, compute_miou, compute_auc = metrics

        train_miou, train_auc = [], []
        train_miou.append(compute_miou(y, pred))
        train_auc.append(compute_auc(y[:, :, :, 0], pred))

        # Training Prints
        tf.print('Step', step, '/', opts.num_steps, 
                 ': loss', loss,
                 ': miou', compute_miou.result(), 
                 ': auc', compute_auc.result())

        with file_writer.as_default():

            image = tf.cast(255 * x, tf.uint8)
            mask = tf.cast(255 * y, tf.uint8)
            summary_predictions = tf.cast(tf.expand_dims(pred * 255, axis=-1), tf.uint8)


            tf.summary.image('Train_image', image, step=tf.cast(step, tf.int64), max_outputs=2)
            tf.summary.image('Train_mask', mask, step=tf.cast(step, tf.int64), max_outputs=2)
            tf.summary.image('Train_prediction', summary_predictions, step=tf.cast(step, tf.int64),
                             max_outputs=2)
            tf.summary.scalar('Training Loss', loss, step=tf.cast(step, tf.int64))

            tf.summary.scalar('Training mIoU', sum(train_miou) / len(train_miou),
                              step=tf.cast(step, tf.int64))
            tf.summary.scalar('Training AUC', sum(train_auc) / len(train_auc), step=tf.cast(step, tf.int64))

            # Logging the optimizer's hyperparameters
            for key in optimizer._hyper:
                tf.summary.scalar(key,optimizer._hyper[key].numpy(),step=tf.cast(step, tf.int64))
            # Extract weights and filter out None elemens for aspp without weights
            if opts.model == 'deeplab':
                weights = filter(None, [x.weights for x in model.layers])
            else:
                weights = filter(None, [x.weights for x in model.layers[0].layers])
            for var in weights:
                tf.summary.histogram('%s' % var[0].name, var[0], step=tf.cast(step, tf.int64))

        file_writer.flush()

        # model.save('model.h5')

    return


def log_validation_step(opts, file_writer, image, mask, step, pred, val_loss, val_miou, val_auc):
    """ Log to file writer after a validation step """
    if hvd.local_rank() == 0 and hvd.rank() == 0:

        with file_writer.as_default():
            tf.summary.image('Validation image', image, step=tf.cast(step, tf.int64), max_outputs=5)
            tf.summary.image('Validation mask', mask, step=tf.cast(step, tf.int64), max_outputs=5)
            tf.summary.image('Validation prediction', pred, step=tf.cast(step, tf.int64), max_outputs=5)
            tf.summary.scalar('Validation Loss', val_loss, step=tf.cast(step, tf.int64))
            tf.summary.scalar('Validation Mean IoU', val_miou, step=tf.cast(step, tf.int64))
            tf.summary.scalar('Validation AUC', val_auc, step=tf.cast(step, tf.int64))

        file_writer.flush()

        tf.print('Validation at step', step, 
                 ': validation loss', val_loss,
                 ': validation miou', val_miou, 
                 ': validation auc', val_auc)

    return
