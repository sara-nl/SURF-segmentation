import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import pdb
import glob
from os import listdir
from openslide import OpenSlide, ImageSlide, OpenSlideUnsupportedFormatError
from PIL import Image, ImageStat, ImageDraw, ImageFont
from xml.etree.ElementTree import parse
from os.path import join, isfile, exists, splitext
import collections
import os
import cv2
import math
import pdb
import multiprocessing
import sys
import argparse


TEST_PATCHES_PATH = '/lustre4/2/managed_datasets/CAMELYON17/testing/patch_size_2048/patient_100_node_0'
PATCH_SIZE = 2048
pred_string_end = '/*.png'

class WSI(object):


    def_level = 7
    mag_factor = pow(2, def_level)

    def read_wsi_tumor(self, wsi_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            self.cur_wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)

            self.rgb_image_pil = self.wsi_image.read_region((0, 0), self.level_used,
                                                            self.wsi_image.level_dimensions[self.level_used])
            self.rgb_image = np.array(self.rgb_image_pil)

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return False

        return True


    def run_on_test_data(self):
        wsi_paths = sorted(glob.glob(TEST_PATCHES_PATH + pred_string_end, recursive=True))

        # model = tf.keras.models.load_model('path_to_saved_model')
        coord_list          = [tuple(map(tuple,np.load(x.replace('2048_','2048_xy_').replace('.png','.npy'))))[0] for x in wsi_paths]
        coord_list_resized  = [(int(x[0] / self.mag_factor), int(x[1] / self.mag_factor)) for x in coord_list]

        # Construct heatmap array
        max_coord           = np.max(np.array(ts))
        heatmap_size        = (max_coord + math.ceil(PATCH_SIZE/self.def_level),max_coord + math.ceil(PATCH_SIZE/self.def_level))
        heatmap             = np.zeros(heatmap_size)

        pred_dataset = tf.data.Dataset.from_tensor_slices((path_list,coord_list_resized))

        pred_dataset = pred_dataset.apply(
            tf.data.experimental.map_and_batch(
                map_func=lambda x, y, z: load_data(image_path=x, mask_path=y, sample_weight=z, augment=False),
                batch_size=BATCH_SIZE,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                drop_remainder=True))

        pred_dataset = pred_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        for x,y in pred_dataset:
            pred = model.predict(x)
            pred_resized = tf.image.resize(pred,(PATCH_SIZE / self.def_level, PATCH_SIZE / self.def_level), method=ResizeMethod.BILINEAR)

            "TODO: get shapes of pred_resized"
            heatmap[y[0]:"shape pred_resized",y[1]:"shape pred_resized"] = pred_resized





"""
Get Metastase for WSI
"""

if __name__ == "__main__":
    wsi = WSI()
    wsi.run_on_test_data()
