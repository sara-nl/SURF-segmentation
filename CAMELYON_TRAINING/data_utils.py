import numpy as np
from glob import glob
import os
from sklearn.utils import shuffle
from PIL import Image
import tensorflow as tf
from joblib import Parallel, delayed
import multiprocessing
import horovod.tensorflow as hvd
import pdb

def get_image_lists(opts):
    """ Get the image lists"""

    if opts.dataset == "17":
        image_list, mask_list, val_image_list, val_mask_list, sample_weight_list = load_camelyon17(opts)
    elif opts.dataset == "16":
        image_list, mask_list, val_image_list, val_mask_list, sample_weight_list = load_camelyon_16(opts)

    print('Found', len(image_list), 'training images')
    print('Found', len(mask_list), 'training masks')
    print('Found', len(val_image_list), 'validation images')
    print('Found', len(val_mask_list), 'validation masks')
    return image_list, mask_list, val_image_list, val_mask_list, sample_weight_list


def load_camelyon_16(opts):
    """  Load the camelyon16 dataset """
    image_list = [x for x in sorted(glob(os.path.join(opts.train_path,'*'), recursive=True)) if 'mask' not in x]
    mask_list = [x for x in sorted(glob(os.path.join(opts.train_path,'*'), recursive=True)) if 'mask' in x]

    image_list, mask_list = shuffle(image_list, mask_list)

    if opts.debug:
        image_list = image_list[0:100]
        mask_list = mask_list[0:100]

    sample_weight_list = [1.0] * len(image_list)

    val_split = int(len(image_list) * (1-opts.val_split))
    val_image_list = image_list[val_split:]
    val_mask_list = mask_list[val_split:]
    sample_weight_list = sample_weight_list[:val_split]
    image_list = image_list[:val_split]
    mask_list = mask_list[:val_split]

    # idx = [np.asarray(Image.open(x))[:, :, 0] / 255 for x in val_mask_list]
    idx = get_valid_idx(val_mask_list)
    num_pixels = opts.img_size ** 2
    valid_idx = [((num_pixels - np.count_nonzero(x)) / num_pixels) >= 0.2 for x in idx]
    valid_idx = [i for i, x in enumerate(valid_idx) if x]

    val_image_list = [val_image_list[i] for i in valid_idx]
    val_mask_list = [val_mask_list[i] for i in valid_idx]

    val_image_list, val_mask_list = shuffle(val_image_list, val_mask_list)

    return image_list, mask_list, val_image_list, val_mask_list, sample_weight_list


def load_camelyon17(opts):
    """ Load the camelyon17 dataset """

    image_list = [x for c in opts.train_centers for x in
                  sorted(glob(os.path.join(opts.train_path.replace('center_XX', f'center_{c}'),'*'), recursive=True)) if
                  'mask' not in x]
    mask_list = [x for c in opts.train_centers for x in
                 sorted(glob(os.path.join(opts.train_path.replace('center_XX', f'center_{c}'),'*') , recursive=True)) if
                 'mask' in x]

    image_list, mask_list = shuffle(image_list, mask_list)

    if opts.debug:
        image_list = image_list[0:100]
        mask_list = mask_list[0:100]

    sample_weight_list = [1.0] * len(image_list)

    # If validating on everything, 00 custom
    if opts.val_centers == [1, 2, 3, 4]:
        val_split = int(len(image_list) * (1-opts.val_split))
        val_image_list = image_list[val_split:]
        val_mask_list = mask_list[val_split:]
        sample_weight_list = sample_weight_list[:val_split]
        image_list = image_list[:val_split]
        mask_list = mask_list[:val_split]

        idx = [np.asarray(Image.open(x))[:, :, 0] / 255 for x in val_mask_list]
        num_pixels = opts.img_size ** 2
        valid_idx = [((num_pixels - np.count_nonzero(x)) / num_pixels) >= 0.2 for x in idx]
        valid_idx = [i for i, x in enumerate(valid_idx) if x]

        val_image_list = [val_image_list[i] for i in valid_idx]
        val_mask_list = [val_mask_list[i] for i in valid_idx]

        val_image_list, val_mask_list = shuffle(val_image_list, val_mask_list)

    else:
        val_image_list = [x for c in opts.val_centers for x in
                          sorted(glob(os.path.join(opts.valid_path.replace('center_XX', f'center_{c}'),'*') , recursive=True)) if
                          'mask' not in x]
        val_mask_list = [x for c in opts.val_centers for x in
                         sorted(glob(os.path.join(opts.valid_path.replace('center_XX', f'center_{c}'),'*') , recursive=True)) if
                         'mask' in x]

        # idx = [np.asarray(Image.open(x))[:, :, 0] / 255 for x in val_mask_list]
        idx = get_valid_idx(val_mask_list)
        num_pixels = opts.img_size ** 2
        valid_idx = [((num_pixels - np.count_nonzero(x)) / num_pixels) >= 0.2 for x in idx]
        valid_idx = [i for i, x in enumerate(valid_idx) if x]

        val_image_list = [val_image_list[i] for i in valid_idx]
        val_mask_list = [val_mask_list[i] for i in valid_idx]

        val_split = int(len(image_list) * opts.val_split)
        val_image_list = val_image_list[:val_split]
        val_mask_list = val_mask_list[:val_split]

        val_image_list, val_mask_list = shuffle(val_image_list, val_mask_list)
    return image_list, mask_list, val_image_list, val_mask_list, sample_weight_list


def get_valid_idx(mask_list):
    """ Get the valid indices of masks by opening images in parallel """
    num_cores = multiprocessing.cpu_count()
    data = Parallel(n_jobs=num_cores)(delayed(open_img)(i) for i in mask_list)
    return data


def open_img(path):
    return np.asarray(Image.open(path))[:, :, 0] / 255


def get_image(image_path, coords=None, img_height=None, img_width=None, mask=False, flip=0, augment=False):
    """Function to load the image (and maybe top left / right coordinates), and possibly mask, and possibly augment the Image

    :param image_path: path to image
    :param coords: (x,y) coordinates of top left corner of image
    :param img_height:
    :param img_width:
    :param mask:
    :param flip:
    :param augment:
    :return:
    """
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
            img = tf.math.divide(img, 255)
            img = tf.math.subtract(tf.math.multiply(2.0, img), 1.0)

    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(tf.image.resize(images=img, size=[
            img_height, img_width]), dtype=tf.int32)
        img = tf.case([
            (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        ], default=lambda: img)
        img = tf.clip_by_value(img, 0, 1)

    if coords == None:
        return img
    else:
        return img, coords


def random_crop(image, mask, img_height=512, img_width=512):
    image_dims = image.shape
    offset_h = tf.random.uniform(
        shape=(1,), maxval=image_dims[0] - img_height, dtype=tf.int32)[0]
    offset_w = tf.random.uniform(
        shape=(1,), maxval=image_dims[1] - img_width, dtype=tf.int32)[0]

    image = tf.image.crop_to_bounding_box(image,
                                          offset_height=offset_h,
                                          offset_width=offset_w,
                                          target_height=img_height,
                                          target_width=img_width)
    mask = tf.image.crop_to_bounding_box(mask,
                                         offset_height=offset_h,
                                         offset_width=offset_w,
                                         target_height=img_height,
                                         target_width=img_width)
    return image, mask


def load_data(opts, image_path, mask_path, img_height, img_width, augment=False, to_flip=False):

    if to_flip:
        flip = tf.random.uniform(
            shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    else:
        flip = 0

    image = get_image(image_path, img_height=img_height, img_width=img_width, flip=flip, augment=augment)
    mask = get_image(mask_path, img_height=img_height, img_width=img_width, mask=True, flip=flip)

    if opts.random_crop:
        image, mask = random_crop(image, mask, img_height=img_height, img_width=img_width)
    return image, mask  # , sample_weight


def get_train_and_val_dataset(opts, image_list=None, mask_list=None,
                              val_image_list=None, val_mask_list=None):
    """ Get the training and validation dataset. Optionally pass the lists of where to find images and masks
        If these are not given, get these first """

    if image_list is None:
        image_list, mask_list, val_image_list, val_mask_list, sample_weight_list = get_image_lists(opts)

    train_dataset = tf.data.Dataset.from_tensor_slices((image_list,mask_list))
    if opts.horovod:
        # let every worker read unique part of dataset
        train_dataset = train_dataset.shard(hvd.size(), hvd.rank())

    train_dataset = train_dataset.shuffle(buffer_size=6000)
    train_dataset = train_dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=lambda x, y: load_data(opts, image_path=x, mask_path=y, img_height=opts.img_size,
                                            img_width=opts.img_size, augment=False),
            batch_size=opts.batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            drop_remainder=True))
    # train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_mask_list))
    if opts.horovod:
        # let every worker read unique part of dataset
        val_dataset = val_dataset.shard(hvd.size(), hvd.rank())

    val_dataset = val_dataset.shuffle(buffer_size=100)
    val_dataset = val_dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=lambda x, y: load_data(opts, image_path=x, mask_path=y, img_height=opts.img_size,
                                            img_width=opts.img_size, augment=False),
            batch_size=opts.batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            drop_remainder=True))
    # val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset
