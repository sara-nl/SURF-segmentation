import tensorflow as tf

import numpy as np
import pdb
import glob
import sys
import time
from openslide import OpenSlide, ImageSlide, OpenSlideUnsupportedFormatError
from PIL import Image, ImageStat, ImageDraw, ImageFont
from xml.etree.ElementTree import parse
from os.path import join, isfile, exists, splitext
import collections
import os
import cv2
import math
import data_utils
import multiprocessing
import argparse


TEST_PATCHES_PATH = '/nfs/managed_datasets/CAMELYON17/testing/patch_size_2048/'
PATCH_SIZE = 2048
BATCH_SIZE = 2


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

        print("Start Loading model...")
        t1 = time.time()
        model = tf.keras.models.load_model('/home/rubenh/examode/deeplab/CAMELYON_TRAINING/saved_model.h5')
        print(f"Loaded model in {time.time() - t1} seconds")

        diagnose_list = []
        for patient in range(self.eval_patient[0], self.eval_patient[1]):
            for node in range(self.eval_node[0],self.eval_node[1]):
                wsi_paths           = [path for path in self.wsi_paths if path.find(f'patient_{patient}_node_{node}') > -1]
                coord_file_list     = glob.glob(str(wsi_paths[0]) + '/*.npy')
                patch_file_list     = glob.glob(str(wsi_paths[0]) + '/*.png')
                coord_list          = [tuple(map(tuple,np.load(x)))[0] for x in coord_file_list]
                coord_list_resized  = [(int(x[0] / self.mag_factor), int(x[1] / self.mag_factor)) for x in coord_list]
                # Construct heatmap array
                max_coord           = max(max(coord_list_resized)[0],max(coord_list_resized)[1])
                heatmap_size        = (max_coord + math.ceil(self.PATCH_SIZE/self.def_level),max_coord + math.ceil(self.PATCH_SIZE/self.def_level))
                heatmap             = np.zeros(heatmap_size)

                pred_dataset = tf.data.Dataset.from_tensor_slices((patch_file_list,coord_list_resized))

                pred_dataset = pred_dataset.apply(
                    tf.data.experimental.map_and_batch(
                        map_func=lambda x, y: data_utils.get_image(x, coords=y, img_height=self.PATCH_SIZE, img_width=self.PATCH_SIZE),
                        batch_size=self.BATCH_SIZE,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        drop_remainder=True))

                pred_dataset = pred_dataset.prefetch(tf.data.experimental.AUTOTUNE)

                for x,y in pred_dataset:
                    x = tf.image.resize(x, [1024, 1024])
                    pred = model.predict(x)
                    pred =  tf.argmax(pred,axis=-1)
                    self.PATCH_SIZE = 1024
                    pred_resized = tf.image.resize(pred[...,tf.newaxis],[int(self.PATCH_SIZE / self.def_level), int(self.PATCH_SIZE / self.def_level)])
                    heatmap[y[0][0]:y[0][0] + pred_resized.shape[1], y[1][0]:y[1][0] + pred_resized.shape[2]] = pred_resized[0, :, :, 0]*255
                    heatmap[y[0][1]:y[0][1] + pred_resized.shape[1], y[1][1]:y[1][1] + pred_resized.shape[2]] = pred_resized[0, :, :, 0]*255


                heatmap = heatmap.astype('uint8')
                contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                diameter_tumors = []
                if contours:
                    for c in contours:
                        center, radius = cv2.minEnclosingCircle(c)
                        diameter_tumors.append(2*radius)

                    max_diameter_tumor = max(diameter_tumors)
                else:
                    max_diameter_tumor = 0

                diagnose_list.append((f'patient_{patient}_node_{node}',max_diameter_tumor))
                pdb.set_trace()

    def make_csv(self,diagnose_list):
        metastases = []
        for patient in diagnose_list:
            if patient[1] > 2:
                metastases.append((patient,'macro'))
            elif patient[1] > 0.2 and patient[1] < 2:
                metastases.append((patient, 'micro'))
            elif patient[1] < 0.2:
                metastases.append((patient, 'itc'))
            elif patient[1] == 0:
                metastases.append((patient, 'negative'))

        pNstages = []
        for cases in metastases:
            patient = cases[0][cases.find('patient'):11]
            case = [x for x in cases if cases[0][:11] == patient]
            cnt_neg = [x[1] for x in case].count('negative')
            cnt_mac = [x[1] for x in case].count('macro')
            cnt_mic = [x[1] for x in case].count('micro')
            cnt_itc = [x[1] for x in case].count('itc')
            if cnt_neg == 5:
                pNstages.append((patient+'.zip','pN0'))
            if cnt_itc == 5:
                pNstages.append((patient + '.zip', 'pN0(i+)'))
            if cnt_mic > 0 and cnt_mac ==0:
                pNstages.append((patient + '.zip', 'pN1mi'))
            if (cnt_mic > 0 and cnt_mac > 0) and (cnt_mic + cnt_mac <= 3):
                pNstages.append((patient + '.zip', 'pN1'))
            if (cnt_mic > 0 and cnt_mac > 0) and (cnt_mic + cnt_mac > 3):
                pNstages.append((patient + '.zip', 'pN2'))
            pNstages.append(case)
            metastases = [x for x in metastases if cases[0][:11] != patient]

        import csv
        with open('evaluation.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(pNstages)

        return 1





if __name__ == "__main__":
    wsi = WSI()
    wsi.run_on_test_data()


























