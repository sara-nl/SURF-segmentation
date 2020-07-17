import tensorflow as tf
import horovod.tensorflow as hvd
import sys
import os
import pdb
sys.path.insert(0, os.path.join(os.getcwd(), 'keras-deeplab-v3-plus-master'))
from model import Deeplabv3


def init(opts):
    """ Run initialisation options"""

    print("Now hvd.init")
    if opts.horovod:
        hvd.init()
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        if opts.cuda:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print("hvd.size() = ", hvd.size())
            print("GPU's", gpus, "with Local Rank", hvd.local_rank())
            print("GPU's", gpus, "with Rank", hvd.rank())

            if gpus:
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank() % 4], 'GPU')
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Past hvd.init()")


def get_model_and_optimizer(opts):
    """ Load the model and optimizer """

    model = Deeplabv3(input_shape=(opts.img_size, opts.img_size, 3), classes=2, backbone='xception',opts=opts)

    if opts.horovod:
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if opts.fp16_allreduce else hvd.Compression.none

        opt = tf.optimizers.Adam(0.0001 * hvd.size(), epsilon=1e-1)
        # Horovod: add Horovod DistributedOptimizer.
    else:
        opt = tf.optimizers.Adam(0.0001, epsilon=1e-1)

    print("Compiling model...")

    model.build(input_shape=(opts.img_size, opts.img_size, 3))
    # Setting L2 regularization
    # for layer in model.layers:
    #     if layer.name.find('bn') > -1:
    #         layer.trainable = False
    #     print(layer.name, " Trainable: ", layer.trainable)
    #     if hasattr(layer,'kernel_regularizer'):
    #         layer.kernel_regularizer = tf.keras.regularizers.l2(l=1e-4)
    #         print("      Reg: ",layer.kernel_regularizer )
    return model, opt, compression


def filter_fn(image, mask):
    """ Filter images for images with tumor and non tumor """
    return tf.math.zero_fraction(mask) >= 0.2


def setup_logger(opts):
    """ Setup the tensorboard writer """
    # Sets up a timestamped log directory.
    logdir = f'{opts.log_dir}-{str(opts.img_size)}' 
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    if opts.horovod:
        # Creates a file writer for the log directory.
        if hvd.local_rank() == 0 and hvd.rank() == 0:
            file_writer = tf.summary.create_file_writer(logdir)
        else:
            file_writer = None
    else:
        # If running without horovod
        file_writer = tf.summary.create_file_writer(logdir)

    return file_writer


def log_training_step(opts, model, file_writer, x, y, loss, pred, step, metrics):
    """ Log to file writer during training"""
    if hvd.local_rank() == 0 and hvd.rank() == 0:

        compute_loss, compute_accuracy, compute_miou, compute_auc = metrics

        train_accuracy, train_miou, train_auc = [], [], []
        train_accuracy.append(compute_accuracy(y, pred))
        train_miou.append(compute_miou(y, pred))
        train_auc.append(compute_auc(y[:, :, :, 0], pred))

        # Training Prints
        tf.print('Step', step, '/', opts.num_steps, ': loss', loss, ': accuracy', compute_accuracy.result(),
                 ': miou', compute_miou.result(), ': auc', compute_auc.result())

        with file_writer.as_default():

            image = tf.cast(255 * x, tf.uint8)
            mask = tf.cast(255 * y, tf.uint8)
            summary_predictions = tf.cast(tf.expand_dims(pred * 255, axis=-1), tf.uint8)


            tf.summary.image('Train_image', image, step=tf.cast(step, tf.int64), max_outputs=2)
            tf.summary.image('Train_mask', mask, step=tf.cast(step, tf.int64), max_outputs=2)
            tf.summary.image('Train_prediction', summary_predictions, step=tf.cast(step, tf.int64),
                             max_outputs=2)
            tf.summary.scalar('Training Loss', loss, step=tf.cast(step, tf.int64))
            tf.summary.scalar('Training Accuracy', sum(train_accuracy) / len(train_accuracy),
                              step=tf.cast(step, tf.int64))
            tf.summary.scalar('Training mIoU', sum(train_miou) / len(train_miou),
                              step=tf.cast(step, tf.int64))
            tf.summary.scalar('Training AUC', sum(train_auc) / len(train_auc), step=tf.cast(step, tf.int64))

            # Extract weights and filter out None elemens for aspp without weights
            weights = filter(None, [x.weights for x in model.layers])
            for var in weights:
                tf.summary.histogram('%s' % var[0].name, var[0], step=tf.cast(step, tf.int64))

        file_writer.flush()

        model.save('model.h5')

    return


def log_validation_step(opts, file_writer, image, mask, step, pred, val_loss, val_acc, val_miou, val_auc):
    """ Log to file writer after a validation step """
    if hvd.local_rank() == 0 and hvd.rank() == 0:

        with file_writer.as_default():
            tf.summary.image('Validation image', image, step=tf.cast(step, tf.int64), max_outputs=5)
            tf.summary.image('Validation mask', mask, step=tf.cast(step, tf.int64), max_outputs=5)
            tf.summary.image('Validation prediction', pred, step=tf.cast(step, tf.int64), max_outputs=5)
            tf.summary.scalar('Validation Loss', val_loss, step=tf.cast(step, tf.int64))
            tf.summary.scalar('Validation Accuracy', val_acc, step=tf.cast(step, tf.int64))
            tf.summary.scalar('Validation Mean IoU', val_miou, step=tf.cast(step, tf.int64))
            tf.summary.scalar('Validation AUC', val_auc, step=tf.cast(step, tf.int64))

        file_writer.flush()

        tf.print('Validation at step', step, ': validation loss', val_loss, ': validation accuracy', val_acc,
                 ': validation miou', val_miou, ': validation auc', val_auc)

    return
