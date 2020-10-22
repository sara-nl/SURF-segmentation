import numpy as np
from glob import glob
import os
from sklearn.utils import shuffle
from PIL import Image, ImageStat, ImageDraw, ImageFont
import tensorflow as tf
import horovod.tensorflow as hvd
import pdb
import pyvips
import cv2
import sys
import random
import time
import difflib
import itertools
from openslide import OpenSlide, ImageSlide, OpenSlideUnsupportedFormatError
import logging
import hashlib
import io
import json
import multiprocessing
import os
import math
import xml.etree.ElementTree as ET
import numpy as np
import PIL.Image

from utils import rank00

sys.path.insert(0, '$PROJECT_DIR/xml-pathology')


# sys.path.insert(0,'~/SURF-deeplab/TRAINING/xml-pathology')
# from xmlpathology.batchgenerator.utils import create_data_source
# from xmlpathology.batchgenerator.generators import BatchGenerator
# from xmlpathology.batchgenerator.core.samplers import LabelSamplerLoader, SamplerLoader
# from xmlpathology.batchgenerator.core.samplers import SegmentationLabelSampler, Sampler
# from xmlpathology.batchgenerator.callbacks import OneHotEncoding, FitData

def rank00():
    if hvd.rank() == 0 and hvd.local_rank() == 0:
        return True

# PyVips Conversion
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


class PreProcess():
    def __init__(self,opts):
        self.opts = opts

    def _load(image,mask,augment=False):
        if augment:
            img = tf.image.random_brightness(image, max_delta=50.)
            img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
            img = tf.image.random_hue(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        else:
            # img = tf.clip_by_value(img, 0, 2)
            img = tf.math.divide(image, 255)
            img = tf.math.subtract(tf.math.multiply(2.0, img), 1.0)

        mask = tf.clip_by_value(tf.cast(mask,tf.int32), 0, 1)

        return img,mask

    def tfdataset(self,x,y):
        dataset = tf.data.Dataset.from_tensor_slices((x,y))
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=lambda im, msk: PreProcess._load(im,msk,augment=False),
            batch_size=self.opts.batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            drop_remainder=True))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
        self.valid_paths = [('/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor/Tumor_055.tif',
                             '/nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/Mask/Tumor_055_Mask.tif')]

class RadSampler():
    """


    """

    def __init__(self, opts):
        super().__init__()

        self.batch_size = opts.batch_size
        self.patch_size = opts.img_size
        self.tumor_ratio = opts.batch_tumor_ratio
        self.slide_format = opts.slide_format
        self.label_format = opts.label_format
        self.valid_slide_format = opts.valid_slide_format
        self.valid_label_format = opts.valid_label_format
        self.cpus = opts.sample_processes
        self.resolution = [opts.resolution]
        self.label_map = opts.label_map
        self.opts = opts

        datasource_train = create_data_source(data_folder=opts.slide_path,
                                              annotations_path=opts.label_path,
                                              images_extension='.' + self.slide_format,
                                              annotations_extension='.' + self.label_format,
                                              mode='training')

        if rank00(): print(f"Found {len(datasource_train['training'])} training images")
        datasource_validation = create_data_source(data_folder=opts.valid_slide_path,
                                                   annotations_path=opts.valid_label_path,
                                                   images_extension='.' + self.valid_slide_format,
                                                   annotations_extension='.' + self.valid_label_format,
                                                   mode='validation')
        if rank00(): print(f"Found {len(datasource_validation['validation'])} validation images")

        # initialize batchgenerator
        if rank00(): print("Starting Training Batch Generator...")
        self.batchgen_train = BatchGenerator(data_sources=datasource_train,
                                             label_map=self.label_map,
                                             batch_size=self.batch_size,
                                             cpus=self.cpus,
                                             sampler_loader=SamplerLoader(class_=Sampler, input_shapes=[
                                                 [self.patch_size, self.patch_size, 3]],
                                                                          spacings=self.resolution),
                                             label_sampler_loader=LabelSamplerLoader(class_=SegmentationLabelSampler),
                                             log_path=opts.log_dir)
        self.batchgen_train.start()

        if rank00(): print("Starting Validation Batch Generator...")
        self.batchgen_validation = BatchGenerator(data_sources=datasource_validation,
                                                  label_map=self.label_map,
                                                  batch_size=self.batch_size,
                                                  cpus=self.cpus,
                                                  sampler_loader=SamplerLoader(class_=Sampler,
                                                                               input_shapes=[
                                                                                   [self.patch_size, self.patch_size,
                                                                                    3]],
                                                                               spacings=self.resolution),
                                                  label_sampler_loader=LabelSamplerLoader(
                                                      class_=SegmentationLabelSampler),
                                                  log_path=opts.log_dir)
        self.batchgen_validation.start()

    def save_image(self, sample, train=True):

        wsi_name = sample['auxiliaries']['sampler'][0]['sample_info'][0]['image']
        wsi = OpenSlide(wsi_name)
        cx, cy = sample['auxiliaries']['sampler'][0]['sample_info'][0]['center']
        pil = wsi.get_thumbnail(wsi.level_dimensions[-2])
        mag = len(wsi.level_dimensions) - 2
        cx, cy = int(cx / 2 ** mag), int(cy / 2 ** mag)

        if train:
            im_name = os.path.join(self.opts.log_dir, 'save_train' + wsi_name.split('/')[-1])
        else:
            im_name = os.path.join(self.opts.log_dir, 'save_valid' + wsi_name.split('/')[-1])

        size = int(self.patch_size / mag / 2)
        if os.path.isfile(im_name):
            im = np.array(Image.open(im_name))
            im[cy - size:cy + size, cx - size:cx + size, 0] = 0
            im[cy - size:cy + size, cx - size:cx + size, 1] = 255
            im[cy - size:cy + size, cx - size:cx + size, 2] = 0
        else:
            im = np.array(pil)
            im[cy - size:cy + size, cx - size:cx + size, 0] = 0
            im[cy - size:cy + size, cx - size:cx + size, 1] = 255
            im[cy - size:cy + size, cx - size:cx + size, 2] = 0

        Image.fromarray(im).save(im_name)

        return

    def get_next(self, train=True):

        # Every new iteration, new sample
        if train:
            # Ugly, but rescaling in place
            sample = self.batchgen_train.batch('training')
            im = (sample['x_batch'] * 255).astype('float32')
            msk = sample['y_batch'][..., None].astype('float32')
            print(msk.max())
            RadSampler.save_image(self, sample, train=train)

            dataset = tf.data.Dataset.from_tensor_slices((im, msk))
            dataset = dataset.apply(
                tf.data.experimental.map_and_batch(
                    map_func=lambda im, msk: RadSampler._load(im, msk, augment=False),
                    batch_size=self.batch_size,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    drop_remainder=True))
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            # Ugly, but rescaling in place
            sample = self.batchgen_train.batch('validation')
            im = (sample['x_batch'] * 255).astype('float32')
            msk = sample['y_batch'][..., None].astype('float32')
            RadSampler.save_image(self, sample, train=train)

            dataset = tf.data.Dataset.from_tensor_slices((im, msk))
            dataset = dataset.apply(
                tf.data.experimental.map_and_batch(
                    map_func=lambda im, msk: RadSampler._load(im, msk, augment=False),
                    batch_size=self.batch_size,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    drop_remainder=True))
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class SurfSampler(tf.keras.utils.Sequence):
    """
    - This sampler samples patches from whole slide images  in several formats, from
    which it samples the patch on the WSI and the same patch on the WSI mask.

    !! This Sampler needs to be used with OpenSlide and PyVips library installed
    in the backend, see https://github.com/sara-nl/SURF-deeplab/blob/master/README.md

    - Furthermore it needs to glob over directories that have the following structure:

    ---`opts.slide_path`/
                        WSI_001.`opts.slide_format`
                        WSI_002.`opts.slide_format`
                        ...

    ---`opts.label_path`/
                        WSI_Mask_001.`opts.label_format`
                        WSI_Mask_002.`opts.label_format`
                        ...


    !! Label and WSI's are matched on string similarity (https://docs.python.org/3/library/difflib.html -> difflib.get_close_matches() )

    - It samples a batch according to `opts.batch_size`, with the batch
    consisting of patches that contain tumor and non - tumor, based on
    `opts.batch_tumor_ratio` \in [0,1] (rounded to ints)

    - It samples out of contours made with OpenCV thresholding

    - Furthermore it contains a hard-coded standard deviation threshold, which
    can discard patches if not above some stddev. This is to avoid sampling
    patches that are background. From experience on CAMELYON16/17 this works
    as intended, no guarantees are given for other datasets

   >>>>Example:

   mpirun -map-by ppr:4:node -np 4 -x LD_LIBRARY_PATH -x PATH python -u train.py \
    --img_size 1024 \
    --horovod \
    --batch_size 2 \
    --fp16_allreduce \
    --log_dir /home/rubenh/examode/deeplab/CAMELYON_TRAINING/logs/test/ \
    --log_every 2 \
    --num_steps 5000 \
    --slide_format tif \
    --mask_format tif \
    --slide_path /nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor \
    --mask_path /nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/Mask \
    --bb_downsample 7 \
    --batch_tumor_ratio 0.5 \
    --log_image_path logs/test/

    """
    def __init__(self, opts, training=True, evaluate=False):
        super().__init__()

        slides = sorted(glob(os.path.join(opts.slide_path,f'*.{opts.slide_format}')))
        labels = sorted(glob(os.path.join(opts.label_path,f'*.{opts.label_format}')))

        # Match labels to slides (all slides must have labels)
        self.train_paths = shuffle([(difflib.get_close_matches(label.split('/')[-1].split('.')[-2],slides,n=1,cutoff=0.1)[0],label) for label in labels])

        # Custom path removal for Camelyon 17
        if opts.slide_path.find('CAMELYON17') > 0:
            _del = []
            for data in self.train_paths:
                if data[0].split('/')[-1].split('.')[-2] != data[1].split('/')[-1].split('.')[-2]:
                    _del.append(data)

            self.train_paths = [data for data in self.train_paths if data not in _del]

        if rank00(): print(f"\nFound {len(self.train_paths)} slides")
        if opts.valid_slide_path:
            valid_slides = glob(os.path.join(opts.valid_slide_path, f'*.{opts.slide_format}'))
            valid_labels = glob(os.path.join(opts.valid_label_path, f'*.{opts.label_format}'))

            # Match labels to slides (all slides must have labels)
            self.valid_paths = shuffle(
                [(difflib.get_close_matches(label.split('/')[-1], valid_slides, n=1, cutoff=0.1)[0], label) for label in
                 valid_labels])
        else:
            val_split = int(len(self.train_paths) * (1 - opts.val_split))
            self.valid_paths = self.train_paths[val_split:]
            self.train_paths = self.train_paths[:val_split]

        if rank00(): print(f"\nWith {len(self.train_paths)} slides")
        if rank00(): print(f"and {len(self.valid_paths)} validation/test slides\n")

        self.batch_size = opts.batch_size
        self.contours_train = []
        self.contours_valid = []
        self.contours_tumor = []
        self.pixelpoints    = []
        self.save_data      = []
        self.level_used = opts.bb_downsample
        self.mag_factor = pow(2, self.level_used)
        self.patch_size = opts.img_size
        self.tumor_ratio = opts.batch_tumor_ratio
        self.log_image_path = opts.log_dir
        self.slide_format = opts.slide_format
        self.evaluate = evaluate
        self.cnt = 0
        self.train = training
        self.verbose = opts.verbose

        # Make sure that every process has at least 1 WSI
        assert hvd.size() <= len(self.valid_paths), "WARNING: {hvd.size()} workers will share {len(self.valid_paths)} images"
        testims = len(self.valid_paths)
        ims_per_worker = testims // hvd.size()
        self.valid_paths = self.valid_paths[hvd.rank()*ims_per_worker:(hvd.rank()+1)*ims_per_worker]
        #self.valid_paths = [('/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor/Tumor_055.tif',
        #                     '/nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/Mask/Tumor_055_Mask.tif')]

    def __len__(self):
        return math.ceil(len(self.train_paths) / self.batch_size)

    def get_bb(self):
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # (50, 50)
        close_kernel = np.ones((50, 50), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        # (30, 30)
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
        contours, _ = cv2.findContours(np.array(image_open), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _offset = 0
        for i, contour in enumerate(contours):
            # sometimes the bounding boxes annotate a very small area not in the ROI
            if contour.shape[0] < 10:
                if rank00() and self.verbose == 'debug': print(f"Deleted too small contour from {self.cur_wsi_path}")
                del contours[i]
                _offset += 1
                i = i - _offset        self.valid_paths = [('/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor/Tumor_055.tif',
                             '/nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/Mask/Tumor_055_Mask.tif')]
        # contours_rgb_image_array = np.array(self.rgb_image)
        # line_color = (255, 150, 150)
        # cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 1)
        # Image.fromarray(contours_rgb_image_array[...,:3]).save('test.png')

        # self.rgb_image_pil.close()
        # self.wsi.close()
        # self.mask.close()

        return contours


    def trainer(self, image, mask_image, img_reg, mask_reg, numpy_batch_patch, numpy_batch_mask, save_image):

        for i in range(int(self.batch_size * (1 - self.tumor_ratio))):
            bc = random.choice(self.contours)
            msk = np.zeros(self.rgb_image.shape, np.uint8)
            cv2.drawContours(msk, [bc], -1, (255), -1)
            pixelpoints = np.transpose(np.nonzero(msk))

            b_x_start = bc[..., 0].min() * self.mag_factor
            b_y_start = bc[..., 1].min() * self.mag_factor
            b_x_end = bc[..., 0].max() * self.mag_factor
            b_y_end = bc[..., 1].max() * self.mag_factor
            h = b_y_end - b_y_start
            w = b_x_end - b_x_start

            patch = []
            k = 0
            while not len(patch):
                x_topleft = random.choice(pixelpoints)[1] * self.mag_factor
                y_topleft = random.choice(pixelpoints)[0] * self.mag_factor
                t1 = time.time()
                # if trying to fetch outside of image, retry
                try:
                    patch = img_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
                    patch = np.ndarray((self.patch_size, self.patch_size, image.get('bands')), buffer=patch,
                                       dtype=np.uint8)[..., :3]
                    _std = ImageStat.Stat(Image.fromarray(patch)).stddev
                    k += 1

                    # discard based on stddev
                    if k < 5:
                        if (sum(_std[:3]) / len(_std[:3])) < 15:
                            if self.verbose == 'debug':
                                print("Discard based on stddev")
                            patch = []
                except Exception as e:
                    print("Exception in extracting patch: ", e)
                    patch = np.random.normal(size=(self.patch_size,self.patch_size))

            if rank00() and self.verbose == 'debug':
                print(f"Sample {self.patch_size} x {self.patch_size} from contours = {h}" + f" by {w} in {time.time() -t1} seconds")
            numpy_batch_patch.append(patch)

            mask  = mask_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
            mask  = np.ndarray((self.patch_size,self.patch_size,mask_image.get('bands')),buffer=mask, dtype=np.uint8)
            numpy_batch_mask.append(mask)
            x,y,imsize = x_topleft, y_topleft, self.patch_size
            coords = list(itertools.product(list(range(y,y+imsize)),list(range(x,x+imsize))))

            try:
                # Draw the rectangles of sampled images on downsampled rgb
                save_image = cv2.drawContours(save_image, self.contours, -1, (0,255,0), 1)
                # save_image = cv2.rectangle(save_image, (int(x_topleft // self.mag_factor) , int(y_topleft // self.mag_factor)),
                #                                        (int((x_topleft + self.patch_size) // self.mag_factor), int((y_topleft + self.patch_size) // self.mag_factor)),
                #                                        (255,255,255), 2)
            except:
                pass

            self.save_data.append(({'patch'      : patch,
                                    'image'      : save_image,
                                    'file_name'  : self.cur_wsi_path[0],
                                    'coords'     : coords,
                                    'mask'       : mask,
                                    'tumor'      : 1}))


        for i in range(int(self.batch_size * (self.tumor_ratio))):

            bc = random.choice(self.contours_tumor)
            msk = np.zeros(self.rgb_image.shape, np.uint8)

            cv2.drawContours(msk, [bc], -1, (255), -1)
            pixelpoints = np.transpose(np.nonzero(msk))

            b_x_start = bc[..., 0].min() * self.mag_factor
            b_y_start = bc[..., 1].min() * self.mag_factor
            b_x_end = bc[..., 0].max() * self.mag_factor
            b_y_end = bc[..., 1].max() * self.mag_factor
            h = b_y_end - b_y_start
            w = b_x_end - b_x_start

            patch = []

            k = 0
            while not len(patch):
                x_topleft = random.choice(pixelpoints)[1] * self.mag_factor
                y_topleft = random.choice(pixelpoints)[0] * self.mag_factor
                t1 = time.time()
                # if trying to fetch outside of image, retry
                try:
                    patch = img_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
                    patch = np.ndarray((self.patch_size, self.patch_size, image.get('bands')), buffer=patch,
                                       dtype=np.uint8)[..., :3]
                    _std = ImageStat.Stat(Image.fromarray(patch)).stddev

                    k += 1
                    # discard based on stddev
                    if k < 5:
                        if (sum(_std[:3]) / len(_std[:3])) < 15:
                            if self.verbose == 'debug':
                                print("Discard based on stddev")
                            patch = []

                except Exception as e:
                    print("Exception in extracting patch: ", e)
                    patch = []

            # if rank00(): print(f"Sample {self.patch_size} x {self.patch_size} from contours = {h}" + f" by {w} in {time.time() -t1} seconds")
            numpy_batch_patch.append(patch)

            msk_downsample = 1
            # mask  = mask_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
            mask  = mask_reg.fetch(x_topleft, y_topleft, self.patch_size//msk_downsample, self.patch_size//msk_downsample)
            # mask  = np.ndarray((self.patch_size,self.patch_size,mask_image.get('bands')),buffer=mask, dtype=np.uint8)
            mask  = np.ndarray((self.patch_size//msk_downsample,self.patch_size//msk_downsample,mask_image.get('bands')),buffer=mask, dtype=np.uint8)

            numpy_batch_mask.append(mask)
            x,y,imsize = x_topleft, y_topleft, self.patch_size
            coords = list(itertools.product(list(range(y,y+imsize)),list(range(x,x+imsize))))

            try:
                # Draw the rectangles of sampled images on downsampled rgb
                save_image = cv2.drawContours(save_image, self.contours, -1, (0,255,0), 1)
                # save_image = cv2.rectangle(save_image, (int(x_topleft // self.mag_factor) , int(y_topleft // self.mag_factor)),
                #                                        (int((x_topleft + self.patch_size) // self.mag_factor), int((y_topleft + self.patch_size) // self.mag_factor)),
                #                                        (255,255,255), -1)
            except:
                pass

            self.save_data.append(({   'patch'      : patch,
                                       'image'      : save_image,
                                       'file_name'  : self.cur_wsi_path[0],
                                       'coords'     : coords,
                                       'mask'       : mask,
                                       'tumor'      : 1}))


        try:
            Image.fromarray(save_image[..., :3]).save(os.path.join(self.log_image_path,
                                                                   self.cur_wsi_path[0].split('/')[-1].replace(
                                                                       self.slide_format, 'png')))
        except:
            pass

        return np.array(numpy_batch_patch),np.array(numpy_batch_mask)


    def tester(self,image,mask_image,img_reg,mask_reg,numpy_batch_patch,numpy_batch_mask,save_image):


        for i in range(int(self.batch_size)):
            if not len(self.contours):
                print(f"WARNING: WSI {self.cur_wsi_path[0]} has no contours")
                pass
            bc = self.contours[self.cnt]
            msk = np.zeros(self.rgb_image.shape,np.uint8)
            x_topleft,y_topleft,width,height = cv2.boundingRect(bc)
            cv2.drawContours(msk,[bc],-1,(255),-1)
            # First gather all posssible pixelpoints, then, drop past_coords
            if not len(self.pixelpoints):
                self.pixelpoints = np.transpose(np.nonzero(msk))
                self.pixelpoints = self.pixelpoints[...,:2] * self.mag_factor

            b_x_start = bc[...,0].min() * self.mag_factor
            b_y_start = bc[...,1].min() * self.mag_factor
            b_x_end = bc[...,0].max() * self.mag_factor
            b_y_end = bc[...,1].max() * self.mag_factor
            h = b_y_end - b_y_start
            w = b_x_end - b_x_start

            patch = []

            pixelpoints = pixelpoints * self.mag_factor
            pixelpoints = list(map(tuple, pixelpoints[..., :2]))

            union = set(past_coords) | set(pixelpoints)
            intersect = set(past_coords) & set(pixelpoints)
            non_inter = list(union - intersect)
            pixelpoints = non_inter

            while not len(patch):
                coords = random.choice(self.pixelpoints)
                x_topleft = coords[1]
                y_topleft = coords[0]
                try:

                    patch = img_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
                    patch = np.ndarray((self.patch_size, self.patch_size, image.get('bands')), buffer=patch,
                                       dtype=np.uint8)[..., :3]
                except Exception as e:
                    print("Exception in extracting patch: ", e)
                    patch = []

            # if rank00(): print(f"\n\nTest Sample {self.patch_size} x {self.patch_size} from ROI = {h}" + f" by {w} in {time.time() -t1} seconds\n\n")
            numpy_batch_patch.append(patch)
            msk_downsample = 1

            # mask  = mask_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
            mask  = mask_reg.fetch(x_topleft, y_topleft, self.patch_size//msk_downsample, self.patch_size//msk_downsample)
            # mask  = np.ndarray((self.patch_size,self.patch_size,mask_image.get('bands')),buffer=mask, dtype=np.uint8)
            mask  = np.ndarray((self.patch_size//msk_downsample,self.patch_size//msk_downsample,mask_image.get('bands')),buffer=mask, dtype=np.uint8)

            numpy_batch_mask.append(mask)

            try:
                # Draw the rectangles of sampled images on downsampled rgb
                save_image = cv2.drawContours(save_image, self.contours, -1, (0,255,0), 1)
                save_image = cv2.rectangle(save_image, (int(x_topleft // self.mag_factor) , int(y_topleft // self.mag_factor)),
                                                        (int((x_topleft + self.patch_size) // self.mag_factor), int((y_topleft + self.patch_size) // self.mag_factor)),
                                                        (255,255,255), -1)
            except:
                pass

            x,y,imsize = x_topleft, y_topleft, self.patch_size
            # Get Cartesian product so all patch coordinates are dropped from pixelpoints
            coords = list(itertools.product(list(range(y,y+imsize)),list(range(x,x+imsize))))


            self.save_data.append(({'patch'      : patch,
                                    'image'      : save_image,
                                    'file_name'  : self.cur_wsi_path[0],
                                    'coords'     : coords,
                                    'mask'       : mask,
                                    'tumor'      : 1*(np.count_nonzero(mask) > 0)}))

            # Remove past coordinates from possible pixelpoints
            mask_pixel_in_past_coords=np.isin(self.pixelpoints,coords,invert=True)
            pixel_not_in_past_coords=np.nonzero(mask_pixel_in_past_coords)
            row,co = pixel_not_in_past_coords[0],pixel_not_in_past_coords[1]
            row=row[co>0]
            self.pixelpoints = self.pixelpoints[row,:]

        print(f"\n\nTest sampling at ROI {self.cnt+1} / {len(self.contours)} of {self.cur_wsi_path} with ~ {len(self.pixelpoints) // (self.batch_size*self.patch_size)} iter to go.\n\n")

        # If past all patches of contour, get next contour
        if len(self.pixelpoints) <= self.patch_size:
        # if 1: # for debugging
            self.cnt +=1
            self.pixelpoints = []
            if self.cnt == len(self.contours):
                self.wsi_idx +=1
                self.cnt = 0

        try:
            Image.fromarray(save_image[...,:3]).save(os.path.join(self.log_image_path,self.cur_wsi_path[0].split('/')[-1].replace(self.slide_format,'png')))
        except:
            pass


        return np.array(numpy_batch_patch),np.array(numpy_batch_mask)


    def parse_xml(self,label=None):
        """
            make the list of contour from xml(annotation file)
            input (CAMELYON17):

        <?xml version="1.0"?>
        <ASAP_Annotations>
        	<Annotations>
        		<Annotation Name="Annotation 0" Type="Polygon" PartOfGroup="metastases" Color="#F4FA58">
        			<Coordinates>
        				<Coordinate Order="0" X="12711.2998" Y="88778.1016" />
                        .
                        .
                        .
        			</Coordinates>
        		</Annotation>
        	</Annotations>
        	<AnnotationGroups>
        		<Group Name="metastases" PartOfGroup="None" Color="#ff0000">
        			<Attributes />
        		</Group>
        	</AnnotationGroups>
        </ASAP_Annotations>

            fn_xml = file name of xml file
            downsample = desired resolution
            var:
            li_li_point = list of tumors
            li_point = the coordinates([x,y]) of a tumor
            return  list of list (2D array list)
        """

        li_li_point = []
        tree = ET.parse(label)

        for ASAP_Annotations in tree.getiterator():
            for i_1, Annotations in enumerate(ASAP_Annotations):
                for i_2, Annotation in enumerate(Annotations):
                    for i_3, Coordinates in enumerate(Annotation):
                        li_point = []
                        for i_4, Coordinate in enumerate(Coordinates):
                            x_0 = float(Coordinate.attrib['X'])
                            y_0 = float(Coordinate.attrib['Y'])
                            li_point.append((x_0, y_0))
                        if len(li_point):
                            li_li_point.append(li_point)


        # Make opencv contours
        contours = []
        for li_point in li_li_point:
            li_point_int = [[int(round(point[0])), int(round(point[1]))] for point in li_point]
            contour = np.array(li_point_int, dtype=np.int32)
            contours.append(contour)

        # Make lvl 0 mask
        mask = np.zeros(self.wsi.dimensions,dtype=np.uint8)
        for idx,contour in enumerate(contours):
            cv2.fillPoly(mask, pts =[contour], color=(255))

        return mask

    def __getitem__(self,idx):

        # Every new iteration, new sample
        cnt = 0
        if self.train:
            while not self.contours_train or not self.contours_tumor:
                if cnt > 5: idx+=1
                self.cur_wsi_path = self.train_paths[idx]
                if rank00() and self.verbose == 'debug': print(f"Opening {self.cur_wsi_path}...")

                self.wsi  = OpenSlide(self.cur_wsi_path[0])

                if self.cur_wsi_path[0].find('CAMELYON17') > 0:
                    self.mask =  SurfSampler.parse_xml(self,label=self.cur_wsi_path[1])
                else:
                    self.mask = OpenSlide(self.cur_wsi_path[1])

                self.rgb_image_pil = self.wsi.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                self.rgb_image = np.array(self.rgb_image_pil)
                if self.cur_wsi_path[0].find('CAMELYON17') > 0:
                    self.mask_image = cv2.resize(self.mask,self.wsi.level_dimensions[self.level_used])[...,None]
                else:
                    self.mask_pil = self.mask.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                    self.mask_image = np.array(self.mask_pil)

                self.bounding_boxes_train = self.get_bb()
                self.contours_train = self.get_bb()
                self.contours = self.contours_train

                # Get contours of tumor
                contours, _ = cv2.findContours(self.mask_image[..., 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self.contours_tumor = contours
                cnt += 1

        else:
            while not self.contours_valid or not self.contours_tumor:
                if cnt > 5: idx+=1
                # Drop past WSI's
                self.wsi_idx = idx
                if self.wsi_idx:
                    self.valid_paths = self.valid_paths[self.wsi_idx:]
                    if rank00(): print(f"Evaluated {idx*hvd.size()} / {idx*hvd.size() + len(self.valid_paths)*hvd.size()} test WSI's.")

                self.cur_wsi_path = self.valid_paths[0]
                if rank00() and self.verbose == 'debug': print(f"Opening {self.cur_wsi_path}...")

                self.wsi  = OpenSlide(self.cur_wsi_path[0])
                if self.cur_wsi_path[0].find('CAMELYON17') > 0:
                    self.mask =  SurfSampler.parse_xml(self,label=self.cur_wsi_path[1])
                else:
                    self.mask = OpenSlide(self.cur_wsi_path[1])



                self.rgb_image_pil = self.wsi.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                self.rgb_image = np.array(self.rgb_image_pil)
                if self.cur_wsi_path[0].find('CAMELYON17') > 0:
                    self.mask_image = cv2.resize(self.mask,self.wsi.level_dimensions[self.level_used])[...,None]
                else:
                    self.mask_pil = self.mask.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                    self.mask_image = np.array(self.mask_pil)
                self.bounding_boxes_valid = self.get_bb()

                self.contours_valid = self.get_bb()
                if not self.contours_valid: self.valid_paths.remove(self.cur_wsi_path)
                self.contours = self.contours_valid

                # Get contours of tumor
                contours, _ = cv2.findContours(self.mask_image[...,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                self.contours_tumor = contours if contours else self.contours
                cnt += 1

        image = pyvips.Image.new_from_file(self.cur_wsi_path[0])
        if self.cur_wsi_path[0].find('CAMELYON17') > 0:
            mask_image  = pyvips.Image.new_from_memory(self.mask,self.mask.shape[0],self.mask.shape[1],1,dtype_to_format[str(self.mask.dtype)])
        else:
            mask_image  = pyvips.Image.new_from_file(self.cur_wsi_path[1])

        img_reg = pyvips.Region.new(image)
        mask_reg = pyvips.Region.new(mask_image)

        numpy_batch_patch = []
        numpy_batch_mask = []
        if os.path.isfile(os.path.join(self.log_image_path,
                                       self.cur_wsi_path[0].split('/')[-1].replace(self.slide_format, 'png'))):
            try:
                save_image = np.array(Image.open(os.path.join(self.log_image_path,
                                                              self.cur_wsi_path[0].split('/')[-1].replace(
                                                                  self.slide_format, 'png'))))
            except:
                save_image = self.rgb_image.copy() * np.repeat((self.mask_image + 1)[..., 0][..., np.newaxis], 4,
                                                               axis=-1)
        else:
        # copy image and mark tumor in black
            save_image = self.rgb_image.copy() * np.repeat((self.mask_image + 1)[...,0][...,np.newaxis],4,axis=-1)

        if self.evaluate:
            patches, masks = SurfSampler.tester(self,image,mask_image,img_reg,mask_reg,numpy_batch_patch,numpy_batch_mask,save_image)
        else:
            patches, masks = SurfSampler.trainer(self,image,mask_image,img_reg,mask_reg,numpy_batch_patch,numpy_batch_mask,save_image)

        self.contours_train = []
        self.contours_valid = []
        self.contours_tumor = []

        # print(f"Got item with shape {patches.shape},{masks.shape}")
        return patches.astype('float')/255, masks.astype('float')/255




        return dataset, past_coords, past_wsi, save_data
        self.contours_tumor = []
        return dataset, past_coords, past_wsi, save_data
