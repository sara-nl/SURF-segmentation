import tensorflow as tf

import numpy as np
import pdb
import glob
import sys

sys.path.insert(1,'/home/rubenh/examode/deeplab/CAMELYON_TRAINING')
import data_utils
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
import argparse


TEST_PATCHES_PATH = '/lustre4/2/managed_datasets/CAMELYON17/testing/patch_size_2048/'
PATCH_SIZE = 2048
BATCH_SIZE = 32


class WSI(object):

    def __init__(self):
        self.def_level = 7
        self.mag_factor = pow(2, self.def_level)
        self.PATCH_SIZE = PATCH_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        # self.pred_string_end = '/*.png'
        self.eval_patient = (100,199)
        self.eval_node = (0, 4)

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
        self.wsi_paths = sorted(glob.glob(TEST_PATCHES_PATH + '*', recursive=True))

        """
        TODO:
        
        model = tf.keras.models.load_model('path_to_saved_model')
        
        """
        diagnose_list = []
        for patient in range(self.eval_patient[0], self.eval_patient[1]):
            for node in range(self.eval_node[0],self.eval_node[1]):
                wsi_paths           = [path for path in self.wsi_paths if path.find(f'patient_{patient}_node_{node}') > -1]
                coord_list          = [tuple(map(tuple,np.load(x.replace(f'{self.PATCH_SIZE}_',f'{self.PATCH_SIZE}_xy_').replace('.png','.npy'))))[0] for x in wsi_paths]
                coord_list_resized  = [(int(x[0] / self.mag_factor), int(x[1] / self.mag_factor)) for x in coord_list]
                # Construct heatmap array
                max_coord           = max(max(coord_list_resized)[0],max(coord_list_resized)[1])
                heatmap_size        = (max_coord + math.ceil(self.PATCH_SIZE/self.def_level),max_coord + math.ceil(self.PATCH_SIZE/self.def_level))
                heatmap             = np.zeros(heatmap_size)

                pred_dataset = tf.data.Dataset.from_tensor_slices((wsi_paths,coord_list_resized))

                pred_dataset = pred_dataset.apply(
                    tf.data.experimental.map_and_batch(
                        map_func=lambda x, y: data_utils.get_image(x, coords=y, img_height=self.PATCH_SIZE, img_width=self.PATCH_SIZE),
                        batch_size=self.BATCH_SIZE,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        drop_remainder=True))

                pred_dataset = pred_dataset.prefetch(tf.data.experimental.AUTOTUNE)

                for x,y in pred_dataset:
                    pred = model.predict(x)
                    pred_resized = tf.image.resize(pred,(self.PATCH_SIZE / self.def_level, self.PATCH_SIZE / self.def_level), method=ResizeMethod.BILINEAR)

                    heatmap[y[0]:pred_resized.shape[0],y[1]:pred_resized.shape[1]] = pred_resized



                contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                diameter_tumors = []
                if contours:
                    for c in contours:
                        center, radius = cv2.minEnclosingCircle(c)
                        diameter_tumors.append(2*radius)

                max_diameter_tumor = max(diameter_tumors)

                diagnose_list.append((f'patient_{patient}_node_{node}',max_diameter_tumor))


"""
TODO: from diagnose list
construct csv
"""



if __name__ == "__main__":
    wsi = WSI()
    wsi.run_on_test_data()


























