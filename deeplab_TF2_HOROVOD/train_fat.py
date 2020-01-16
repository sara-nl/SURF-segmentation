from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
import timeit
import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.python.client import device_lib
import sys
from tensorflow.keras import applications
from glob import glob
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pdb
from PIL import Image
from datetime import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from tensorflow.python.keras.utils.data_utils import get_file


sys.path.insert(0,'/home/rubenh/projects/deeplab/deeplab_TF2_HOROVOD/keras-deeplab-v3-plus')

print('TensorFlow', tf.__version__)

for size in [704,224,1024]:

    ## Set Variables ## TODO: implement argument parser
    IMG_SIZE                = size
    EPOCHS                  = 30
    HOROVOD                 = True
    NORMALIZED              = False
    H, W                    = IMG_SIZE, IMG_SIZE  # size of crop
    num_classes             = 1
    RANDOM_CROP             = False
    FLIP                    = False
    POSITIVE_PIXEL_WEIGHT   = 1
    NEGATIVE_PIXEL_WEIGHT   = 1
    fp16_ALLREDUCE          = True
    CUDA                    = False

    if IMG_SIZE == 224:
        BATCH_SIZE = 12
    elif IMG_SIZE == 704:
        BATCH_SIZE = 1
    elif IMG_SIZE == 1024:
        BATCH_SIZE = 2

    TRAIN_PATH = '/home/rubenh/projects/deeplab/CAMELYON16_PREPROCESSING/Processed/patch-based-classification_%s/raw-data/train/' % IMG_SIZE
    VALID_PATH = '/home/rubenh/projects/deeplab/CAMELYON16_PREPROCESSING/Processed/patch-based-classification_%s/raw-data/validation/' % IMG_SIZE

    if NORMALIZED:
        TRAIN_PATH = '/home/rubenh/projects/deeplab/CAMELYON16_PREPROCESSING/Histopathology-Stain-Color-Normalization/%s_out/train/' % IMG_SIZE
        VALID_PATH = '/home/rubenh/projects/deeplab/CAMELYON16_PREPROCESSING/Histopathology-Stain-Color-Normalization/%s_out/validation/' % IMG_SIZE



    print("Now hvd.init")
    if HOROVOD:
        hvd.init()

        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        if CUDA:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print("hvd.size() = ",hvd.size())
            print("GPU's", gpus, "with Local Rank", hvd.local_rank())
            print("GPU's", gpus, "with Rank", hvd.rank())

            if gpus:
                 tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Past hvd.init()")
    if RANDOM_CROP:
        try:
            IMG_SIZE > H

        except Exception as e:
            print('WARNING', e, "Crop Size greater than image size")



    from model import Deeplabv3
    model = Deeplabv3(input_shape=(IMG_SIZE, IMG_SIZE, 3), classes=2, backbone='xception')


    if HOROVOD:
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if fp16_ALLREDUCE else hvd.Compression.none

        opt = tf.optimizers.Adam(0.0001 * hvd.size(),epsilon=1e-1)
        # Horovod: add Horovod DistributedOptimizer.

    print("Compiling model...")

    model.build(input_shape=(IMG_SIZE,IMG_SIZE,3))

    # Setting L2 regularization
    for layer in model.layers:
        if layer.name.find('bn') > -1:
            layer.trainable = False
    #     print(layer.name, " Trainable: ", layer.trainable)
    #     if hasattr(layer,'kernel_regularizer'):
    #         layer.kernel_regularizer = tf.keras.regularizers.l2(l=1e-4)
    #         print("      Reg: ",layer.kernel_regularizer )



    # Horovod: write logs on worker 0.
    verbose = 1 if hvd.rank() == 0 else 0

    size = hvd.size()

    print('Preparing training...')



    compute_loss     = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    compute_accuracy = tf.keras.metrics.Accuracy()
    compute_miou     = tf.keras.metrics.MeanIoU(num_classes=2)



    # @tf.function
    def train_one_step(model, opt, x, y, step, EPOCH):

      with tf.GradientTape() as tape:
            logits  = model(x)
            loss    = compute_loss(y, logits)

      # Horovod: add Horovod Distributed GradientTape.
      tape = hvd.DistributedGradientTape(tape,compression=compression)
      grads = tape.gradient(loss, model.trainable_variables)
      # pdb.set_trace()
      # if np.isnan(grads[0].numpy()).any():
      #   print("Nans in gradient layer 0")
      #   print(grads[0].numpy())
      opt.apply_gradients(zip(grads, model.trainable_variables))

      if step+EPOCH == 0:
          hvd.broadcast_variables(model.variables, root_rank=0)
          hvd.broadcast_variables(opt.variables(), root_rank=0)

      pred = tf.argmax(logits,axis=-1)
      compute_accuracy(y, pred)

      return loss


    # Sets up a timestamped log directory.
    logdir = "logs/train_data/" + str(IMG_SIZE) +'-' + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)

    def train(model, optimizer):
      # train_ds = train_dataset
      iterations = 100 #tf.cast(len(image_list) // BATCH_SIZE, tf.int32)
      loss = 0.0
      accuracy = 0.0
      for EPOCH in range(0,EPOCHS):
            step = 0
            x  = tf.keras.utils.HDF5Matrix('Tumor_001.h5', 'image',normalizer=lambda x: 2*(x/255)-1)
            y  = tf.keras.utils.HDF5Matrix('Tumor_001.h5', 'mask', normalizer=lambda x: x/255)

            x = tf.expand_dims(tf.cast(x.data[()][:4096,:4096],tf.float32),axis=0)
            y = tf.expand_dims(tf.cast(y.data[()][:4096,:4096],tf.float32), axis=0)

            loss = train_one_step(model, optimizer, x, y, step, EPOCH)

            if step % 1 == 0:

                if hvd.local_rank() == 0:

                    # Training Prints
                    tf.print('Step', step, '/', iterations, ' of Epoch', EPOCH, ' Rank',
                             hvd.local_rank(), ': loss', loss, '; accuracy', compute_accuracy.result())

                    compute_accuracy.reset_states()
                    # filter images for images with tumor and non tumor


                file_writer.flush()

            step += 1

      return step, loss, accuracy


    # tf.summary.trace_on(graph=True, profiler=True)
    step, loss, accuracy = train(model, opt)
    print('Final step', step, ': loss', loss, '; accuracy', compute_accuracy.result())

