from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
import timeit
import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.python.client import device_lib
import sys
# from horovod.tensorflow.mpi_ops import Average, Sum, Adasum
from tensorflow.keras import applications
from glob import glob
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pdb
from PIL import Image
from datetime import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import csv
import threading
from pprint import pprint

parser = argparse.ArgumentParser(description='TensorFlow DeeplabV3+ model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_size', type=int, default=1024, help='Image size to use')
parser.add_argument('--cam', type=int, default=17, help='Dataset to process')

opts = parser.parse_args()

# opts.batch_size = 2  # Hardcode batch size for now

pprint(vars(opts))

from tensorflow.python.keras.utils.data_utils import get_file
# tf.enable_eager_execution()

sys.path.insert(0,'/home/rubenh/projects/camelyon/deeplab/deeplab_TF2_HOROVOD/keras-deeplab-v3-plus')

print('TensorFlow', tf.__version__)


for val_center in range(1,5):
    print("THIS IS CENTER: ", val_center )
    ## Set Variables ## TODO: implement argument parser
    IMG_SIZE                = opts.img_size
    EPOCHS                  = 20
    HOROVOD                 = True
    NORMALIZED              = False
    # whether validation dataset is separate path
    VALID_DIFF              = True
    H, W                    = IMG_SIZE, IMG_SIZE  # size of crop
    num_classes             = 1
    RANDOM_CROP             = False
    FLIP                    = False
    POSITIVE_PIXEL_WEIGHT   = 1
    NEGATIVE_PIXEL_WEIGHT   = 1
    fp16_ALLREDUCE          = True
    CUDA                    = False
    CAMELYON                = opts.cam

    if IMG_SIZE == 256:
        BATCH_SIZE = 2
    elif IMG_SIZE == 704:
        BATCH_SIZE = 2
    elif IMG_SIZE == 1024:
        BATCH_SIZE = 2
    elif IMG_SIZE == 2048:
        BATCH_SIZE = 1

    if CAMELYON == 17:
        TRAIN_PATH = '/home/rubenh/projects/camelyon/deeplab/CAMELYON17_PREPROCESSING/'
        train_string_end = 'pro_patch_positive_%s/*center_[!%s]*' % (IMG_SIZE,val_center)
        # Dynamic path ends to glob over
        # train_string_end = '[!%s]*/patches_positive_%s/*'%(val_center,IMG_SIZE)
    elif CAMELYON == 16:
        TRAIN_PATH = '/home/rubenh/projects/camelyon/deeplab/CAMELYON16_PREPROCESSING'
        train_string_end = '*/pro_patch_positive_%s/*' % (IMG_SIZE)

    VALID_PATH = '/home/rubenh/projects/camelyon/deeplab/CAMELYON17_PREPROCESSING/pro_patch_positive_%s/*center_%s*'%(IMG_SIZE,val_center)


    if NORMALIZED:
        TRAIN_PATH = '/home/rubenh/projects/deeplab/CAMELYON16_PREPROCESSING/Histopathology-Stain-Color-Normalization/%s_out/train/' % IMG_SIZE
        VALID_PATH = '/home/rubenh/projects/deeplab/CAMELYON16_PREPROCESSING/Histopathology-Stain-Color-Normalization/%s_out/validation/' % IMG_SIZE




    valid_string_end = '*'

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
                 tf.config.experimental.set_visible_devices(gpus[hvd.local_rank() % 4], 'GPU')
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Past hvd.init()")
    if RANDOM_CROP:
        try:
            IMG_SIZE > H
        except Exception as e:
            print('WARNING', e, "Crop Size greater than image size")


    image_list = [x for x in sorted(glob(TRAIN_PATH + train_string_end, recursive=True)) if 'mask' not in x]

    mask_list = [x for x in sorted(glob(TRAIN_PATH  + train_string_end, recursive=True)) if 'mask' in x]

    sample_weight_list = [1]*len(image_list)
    # sample_weight_list     = [ np.mean(np.where(np.asarray(Image.open(x))[:, :, 0] == 0, NEGATIVE_PIXEL_WEIGHT, POSITIVE_PIXEL_WEIGHT)) for x in sorted(glob(TRAIN_PATH + train_string_end, recursive=True)) if 'mask' in x]

    if VALID_DIFF:
        val_image_list = [x for x in sorted(glob(VALID_PATH + valid_string_end, recursive=True)) if 'mask' not in x]
        val_mask_list = [x for x in sorted(glob(VALID_PATH  + valid_string_end, recursive=True)) if 'mask' in x]
        # sample_weight_list_val = [ np.mean(np.where(np.asarray(Image.open(x))[:, :, 0] == 0, NEGATIVE_PIXEL_WEIGHT, POSITIVE_PIXEL_WEIGHT)) for x in sorted(glob(VALID_PATH + valid_string_end, recursive=True)) if 'mask' in x]
        sample_weight_list_val = [1] * len(val_image_list)
    else:

        val_split =  int(len(image_list)*0.85)
        val_image_list = image_list[val_split:]
        val_mask_list  = mask_list[val_split:]
        sample_weight_list = sample_weight_list[:val_split]
        image_list     = image_list[:val_split]
        mask_list      = mask_list[:val_split]

    if NORMALIZED:
        image_list = [x for x in sorted(glob(TRAIN_PATH + '*', recursive=True)) if 'mask' not in x]

        mask_list = [x for x in sorted(glob(TRAIN_PATH + '*', recursive=True)) if 'mask' in x]

        sample_weight_list = [
            np.mean(np.where(np.asarray(Image.open(x))[:, :, 0] == 0, NEGATIVE_PIXEL_WEIGHT, POSITIVE_PIXEL_WEIGHT))
            for x
            in sorted(glob(TRAIN_PATH + '*', recursive=True)) if 'mask' in x]

        val_image_list = [x for x in sorted(glob(VALID_PATH + '*', recursive=True)) if 'mask' not in x]
        val_mask_list = [x for x in sorted(glob(VALID_PATH + '*', recursive=True)) if 'mask' in x]
        # sample_weight_list_val  = [ int(np.mean(((np.asarray(Image.open(x))[:,:,0]-255)).astype(int))*3+1) for x in sorted(glob(VALID_PATH + 'label-1/*',recursive=True)) if 'mask' in x]

    print('Found', len(image_list), 'training images')
    print('Found', len(mask_list), 'training masks')
    print('Found', len(val_image_list), 'validation images')
    print('Found', len(val_mask_list), 'validation masks')


    def get_image(image_path, img_height=IMG_SIZE, img_width=IMG_SIZE, mask=False, flip=0, augment=False):

        img = tf.io.read_file(image_path)
        if not mask:
            if augment:
                img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
                img = tf.image.resize(images=img, size=[img_height, img_width])
                img = tf.image.random_brightness(img, max_delta=50.)
                img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
                img = tf.image.random_hue(img, max_delta=0.2)
                img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
                img = tf.clip_by_value(img, 0, 255)
                img = tf.case([
                    (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
                ], default=lambda: img)
            else:

                img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.int32)
                img = tf.image.resize(images=img, size=[img_height, img_width])
                # img = tf.clip_by_value(img, 0, 2)
                img = tf.math.divide(img,255)
                img = tf.math.subtract(tf.math.multiply(2.0,img),1.0)

        else:
            img = tf.image.decode_png(img, channels=1)
            img = tf.cast(tf.image.resize(images=img, size=[
                img_height, img_width]), dtype=tf.int32)
            img = tf.case([
                (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
            ], default=lambda: img)
            img = tf.clip_by_value(img, 0, 1)
            # img = tf.one_hot(tf.squeeze(img), depth=num_classes)
            # img = tf.math.subtract(tf.math.multiply(2, img), 1)

        return img


    def random_crop(image, mask, H=512, W=512):
        image_dims = image.shape
        offset_h = tf.random.uniform(
            shape=(1,), maxval=image_dims[0] - H, dtype=tf.int32)[0]
        offset_w = tf.random.uniform(
            shape=(1,), maxval=image_dims[1] - W, dtype=tf.int32)[0]

        image = tf.image.crop_to_bounding_box(image,
                                              offset_height=offset_h,
                                              offset_width=offset_w,
                                              target_height=H,
                                              target_width=W)
        mask = tf.image.crop_to_bounding_box(mask,
                                             offset_height=offset_h,
                                             offset_width=offset_w,
                                             target_height=H,
                                             target_width=W)
        return image, mask


    def load_data(image_path, mask_path, sample_weight=1, H=H, W=W, augment=False):
        if FLIP:
            flip = tf.random.uniform(
                shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
        else:
            flip = 0
        try:
            print(image_path, mask_path)
            image, mask = get_image(image_path, flip=flip, augment=augment), get_image(
                mask_path, mask=True, flip=flip)
        except Exception as e:
            print("Skipping decoding: ", e)

            pass

        if RANDOM_CROP:
            image, mask = random_crop(image, mask, H=H, W=W)
        return image, mask#, sample_weight



    train_dataset = tf.data.Dataset.from_tensor_slices((image_list,
                                                        mask_list,
                                                        sample_weight_list))

    train_dataset = train_dataset.shuffle(buffer_size=2000)
    train_dataset = train_dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=lambda x, y, z: load_data(image_path=x, mask_path=y, sample_weight=z, augment=False),
            batch_size=BATCH_SIZE,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            drop_remainder=True))
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    print(train_dataset)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list,
                                                      val_mask_list))

    val_dataset = val_dataset.apply(
        tf.data.experimental.map_and_batch(map_func=lambda x, y: load_data(image_path=x, mask_path=y, augment=False),
                                           batch_size=BATCH_SIZE,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                           drop_remainder=True))
    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)



    from model import Deeplabv3
    model = Deeplabv3(input_shape=(IMG_SIZE, IMG_SIZE, 3), classes=2, backbone='xception')


    if HOROVOD:
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if fp16_ALLREDUCE else hvd.Compression.none

        opt = tf.keras.optimizers.Adam(0.0001 * hvd.size(),epsilon=1e-1)
        # opt = hvd.DistributedOptimizer(opt,compression=compression)
        # Horovod: add Horovod DistributedOptimizer.

    print("Compiling model...")

    model.build(input_shape=(IMG_SIZE,IMG_SIZE,3))

    # Setting L2 regularization
    # for layer in model.layers:
    #     if layer.name.find('bn') > -1:
    #         layer.trainable = False
    #     print(layer.name, " Trainable: ", layer.trainable)
    #     if hasattr(layer,'kernel_regularizer'):
    #         layer.kernel_regularizer = tf.keras.regularizers.l2(l=1e-4)
    #         print("      Reg: ",layer.kernel_regularizer )



    # Horovod: write logs on worker 0.
    verbose = 1 if hvd.rank() == 0 else 0

    print("hvd.size() is: ",hvd.size())

    print('Preparing training...')



    compute_loss     = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    compute_accuracy = tf.keras.metrics.Accuracy()
    compute_miou     = tf.keras.metrics.MeanIoU(num_classes=2)
    compute_auc      = tf.keras.metrics.AUC()



    # @tf.function
    def train_one_step(model, opt, x, y, step, EPOCH):

      with tf.GradientTape() as tape:
            logits  = model(x)
            loss    = compute_loss(y, logits)

      # Horovod: add Horovod Distributed GradientTape.
      tape = hvd.DistributedGradientTape(tape,odevice_dense='/cpu:0',device_sparse='/cpu:0',compression=compression)
      grads = tape.gradient(loss, model.trainable_variables)


      opt.apply_gradients(zip(grads, model.trainable_variables))

      if step+EPOCH == 0:
          hvd.broadcast_variables(model.variables, root_rank=0)
          hvd.broadcast_variables(opt.variables(), root_rank=0)

      pred = tf.argmax(logits,axis=-1)
      compute_accuracy(y, pred)

      return loss



    # Sets up a timestamped log directory.
    logdir = "logs/train_data/" + str(IMG_SIZE) +'- cam' + str(opts.cam) + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)



    def train(model, optimizer):
      train_ds = train_dataset
      iterations = tf.cast(len(image_list) // (BATCH_SIZE * hvd.size()) , tf.int32)
      loss = 0.0
      accuracy = 0.0
      # train for 50 000 steps
      EPOCH = 0

      for step in range(0,50000):

        for x, y in train_ds:

            if step >= 50000:
                print("Finished Patch Size ", IMG_SIZE)
                sys.exit(0)

            loss = train_one_step(model, optimizer, x, y, step, EPOCH)
            if step % 2000 == 0: #10 * BATCH_SIZE * hvd.size()) == 0 and step > 0:
                print("Beginning Validation")
                if hvd.rank() == 0:

                    # Training Prints
                    tf.print('Step', step, '/', iterations, ' of Epoch', EPOCH, ' Image ', IMG_SIZE, ' Rank',
                             hvd.local_rank(), ': loss', loss, '; accuracy', compute_accuracy.result())

                    compute_accuracy.reset_states()
                    # filter images for images with tumor and non tumor


                    def filter_fn(image, mask):
                        return tf.math.zero_fraction(mask) > 0.2

                    for image, label in val_dataset.filter(filter_fn).shuffle(buffer_size=10):

                        val_loss        = []
                        val_accuracy    = []
                        miou            = []
                        auc             = []
                        val_pred_logits   = model(image)
                        val_pred          = tf.math.argmax(val_pred_logits, axis=-1)
                        val_loss.append(compute_loss(label, val_pred_logits))
                        val_accuracy.append(compute_accuracy(label,val_pred))
                        miou.append(compute_miou(label,val_pred))
                        val_pred = tf.expand_dims(val_pred,axis=-1)
                        auc.append(compute_auc(label, val_pred))

                        with file_writer.as_default():
                            # if step == 10:
                            #     tf.summary.trace_export(name="trace_%s_GPU"%IMG_SIZE,step=tf.cast(EPOCH*iterations+step,tf.int64),profiler_outdir=logdir)

                            image       = tf.cast(255 * ((image - 1 ) / 2) , tf.uint8)
                            mask        = tf.cast(255 * label , tf.uint8)
                            summary_predictions = tf.cast( tf.expand_dims(val_pred * 255, axis=-1),tf.uint8)

                            tf.summary.image('Image', image,                                            step=tf.cast(EPOCH*iterations+step,tf.int64),max_outputs=5)
                            tf.summary.image('Mask',  mask ,                                            step=tf.cast(EPOCH*iterations+step,tf.int64),max_outputs=5)
                            tf.summary.image('Prediction', summary_predictions,                         step=tf.cast(EPOCH*iterations+step,tf.int64),max_outputs=5)
                            tf.summary.scalar('Training Loss',loss,                                     step=tf.cast(EPOCH*iterations+step,tf.int64))
                            tf.summary.scalar('Training Accuracy',compute_accuracy.result(),            step=tf.cast(EPOCH*iterations+step,tf.int64))
                            tf.summary.scalar('Validation Loss',sum(val_loss)/len(val_loss),            step=tf.cast(EPOCH*iterations+step,tf.int64))
                            tf.summary.scalar('Validation Accuracy',sum(val_accuracy)/len(val_accuracy),step=tf.cast(EPOCH*iterations+step,tf.int64))
                            tf.summary.scalar('Mean IoU',sum(miou)/len(miou),                           step=tf.cast(EPOCH*iterations+step,tf.int64))
                            tf.summary.scalar('AUC',sum(auc)/len(auc),                                  step=tf.cast(EPOCH*iterations+step,tf.int64))

                            # Extract weights and filter out None elemens for aspp without weights
                            weights = filter(None,[x.weights for x in model.layers])
                            for var in weights:
                                tf.summary.histogram('%s'%var[0].name,var[0],step=tf.cast(EPOCH*iterations+step,tf.int64))


                        # Validation prints
                        tf.print('   ', 'val_loss', sum(val_loss) / len(val_loss),
                                 '; val_accuracy', sum(val_accuracy) / len(val_accuracy),
                                 '; mean_iou', sum(miou) / len(miou))

                # file_writer.flush()

            step += BATCH_SIZE * hvd.size()



        def filter_hard_mining(image, mask):
            pred_logits = model(image)
            pred        = tf.math.argmax(pred_logits, axis=-1)
            # Only select training samples with miou less then 0.95
            return tf.keras.metrics.MeanIoU(num_classes=2)(mask,pred) < 0.95

        train_ds = train_ds.filter(filter_hard_mining)

      return step, loss.numpy(), compute_miou.result().numpy()


    # tf.summary.trace_on(graph=True, profiler=True)

    step, loss, miou = train(model, opt)

    print('Final step', step, ': loss', loss, '; miou', miou)

