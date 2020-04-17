
from os import listdir
from openslide import OpenSlide, ImageSlide, OpenSlideUnsupportedFormatError
from PIL import Image, ImageStat, ImageDraw, ImageFont
from xml.etree.ElementTree import parse
from os.path import join, isfile, exists, splitext
import collections
import glob
import os
import numpy as np
import cv2
import math
import pdb
import multiprocessing
import tensorflow as tf
import sys
import argparse


parser = argparse.ArgumentParser(description='Postprocessing CAMELYON17',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--patch_size', type=int, default=2048,
                    help='Patch size to generate')

parser.add_argument('--num_threads', type=int, default=4,
                    help='Number of threads used for generating patches, parallelize bounding boxes per WSI')

parser.add_argument('--save_png', type=bool, default=False,
                    help='Whether the png images of bounding boxes and saved patches should be written to disk')

parser.add_argument('--data_folder', type=str, default='cart',
                    help='What data folder pre-pend to use, lisa / cart')


opts = parser.parse_args()


num_threads = opts.num_threads
PATCH_SIZE  = opts.patch_size
if opts.data_folder == 'lisa':
    dir = '/nfs'
else:
    dir = '/lustre4/2'

tf_coord = tf.train.Coordinator()



# modify below directory entries as per your local file system
TRAIN_TUMOR_WSI_PATH = f"{dir}/managed_datasets/CAMELYON17/testing/patients/patient_"
PROCESSED_PATCHES_PATH = f"{dir}/managed_datasets/CAMELYON17/testing/patch_size_{PATCH_SIZE}/"

print("Processing Patch Size ", PATCH_SIZE)



class WSI(object):
    """
        # ================================
        # Class to annotate WSIs with ROIs
        # Class to annotate WSIs with ROIs
        # ================================

    """
    index = 0
    # 1 = 40x
    level_output = 0
    negative_patch_index = 0
    positive_patch_index = 0
    wsi_paths = []
    mask_paths = []
    def_level = 7
    key = 0


    def extract_patches_tumor(self, bounding_boxes):
        """
            Extract both, negative patches from Normal area and positive patches from Tumor area

            Save extracted patches to desk as .png image files

            :param bounding_boxes: list of bounding boxes corresponds to detected ROIs
            :return:

        """
        mag_factor = pow(2, self.level_used-self.level_output)

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        rgb_image_pil = Image.fromarray(self.rgb_image)
        d1 = ImageDraw.Draw(rgb_image_pil)

        rgb_image_pilpro = Image.fromarray(self.rgb_image)
        d2 = ImageDraw.Draw(rgb_image_pilpro)


        print("Processing WSI %s"%self.cur_wsi_path)



        for i, bounding_box in enumerate(bounding_boxes):
            # sometimes the bounding boxes annotate a very small area not in the ROI
            if (bounding_box[2] * bounding_box[3]) < 2500:
                print("Skipped too small Bounding Box %s"%self.cur_wsi_path)
                continue
            b_x_start = int(bounding_box[0]) * mag_factor
            b_y_start = int(bounding_box[1]) * mag_factor
            b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
            b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor


            h = int(bounding_box[2]) * mag_factor
            w = int(bounding_box[3]) * mag_factor
            print("Size of bounding box = %s" % h + " by %s" % w)


            patchidx = 0




            for y_left in range(b_y_start, b_y_end, PATCH_SIZE):

                for x_left in range(b_x_start, b_x_end, PATCH_SIZE):

                    patch = self.wsi_image.read_region((x_left, y_left), self.level_output, (PATCH_SIZE, PATCH_SIZE))
                    patch_array = np.array(patch)
                    _std = ImageStat.Stat(patch).stddev

                    orx = int(x_left / mag_factor)
                    ory = int(y_left / mag_factor)
                    orps = int(PATCH_SIZE / mag_factor)

                    d1.rectangle([(orx,ory),(orx + orps,ory+orps)],outline='green',width=1)

                    ## DISCARD BACKGROUND PIXELS ##

                    # discard based on stddev
                    if ( sum(_std[:3]) / len(_std[:3]) ) < 1 :
                        print("Skipped stddev too low, with stddev = ", sum(_std[:3]) / len(_std[:3]))
                        continue

                    #discard based on black pixels
                    gs = cv2.cvtColor(patch_array, cv2.COLOR_BGR2GRAY)
                    black_pixel_cnt_gt = gs.shape[0]*gs.shape[1] - cv2.countNonZero(gs)
                    if black_pixel_cnt_gt > 10:
                        print("Skipped too many black pixels")
                        continue


                    d2.rectangle([(orx,ory),(orx + orps,ory+orps)],outline='green',width=1)

                    patch.save(PROCESSED_PATCHES_PATH  + self.cur_wsi_path[self.cur_wsi_path.rfind('/')+1:-4] + '/'  + str(PATCH_SIZE) + '_' + self.cur_wsi_path[self.cur_wsi_path.rfind('/')+1:-4]+'_' + str(patchidx)+'.png')
                    np.save(PROCESSED_PATCHES_PATH     + self.cur_wsi_path[self.cur_wsi_path.rfind('/')+1:-4] + '/'  + str(PATCH_SIZE)+'_xy_'+ self.cur_wsi_path[self.cur_wsi_path.rfind('/')+1:-4]+'_' + str(patchidx),np.array([[x_left,y_left]]))


                    patchidx += 1
                    patch.close()

        if opts.save_png:
            rgb_image_pil.save('bb_%s.png'%(self.cur_wsi_path[self.cur_wsi_path.rfind('/')+1:]))
            rgb_image_pilpro.save('bbpro_%s.png'%(self.cur_wsi_path[self.cur_wsi_path.rfind('/')+1:]))

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

            # self.level_used = min(self.def_level, self.wsi_image.level_count - 1, self.mask_image.level_count - 1)
            self.level_used = self.def_level

            self.rgb_image_pil = self.wsi_image.read_region((0, 0), self.level_used, self.wsi_image.level_dimensions[self.level_used])
            self.rgb_image = np.array(self.rgb_image_pil)

            save_path_wsi =  self.cur_wsi_path[:self.cur_wsi_path.find('patients')] + 'patch_size_%s/'%PATCH_SIZE + self.cur_wsi_path[self.cur_wsi_path.rfind('/')+1:-4]

            if not os.path.exists(save_path_wsi):
                os.mkdir(save_path_wsi)

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return False

        return True


    def find_roi_n_extract_patches_tumor(self):
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
        contour_rgb, bounding_boxes = self.get_image_contours_tumor(np.array(image_open), self.rgb_image)


        # pdb.set_trace()
        # Image.fromarray(np.array(contour_rgb)).show()
        # contour_rgb = cv2.resize(contour_rgb, (0, 0), fx=0.40, fy=0.40)
        # cv2.imshow('contour_rgb', np.array(contour_rgb))
        self.rgb_image_pil.close()
        self.extract_patches_tumor(bounding_boxes)
        self.wsi_image.close()

    @staticmethod
    def get_image_contours_tumor(cont_img, rgb_image):
        contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        contours_rgb_image_array = np.array(rgb_image)

        cv2.drawContours(contours_rgb_image_array, contours, -1, (255,150,150), 1)
        # pdb.set_trace()
        # cv2.drawContours(mask_image, contours_mask, -1, line_color, 3)
        return contours_rgb_image_array, bounding_boxes


def run_on_tumor_data():
    wsi.wsi_paths = glob.glob(TRAIN_TUMOR_WSI_PATH + '*1[2-9]*.tif')
    wsi.wsi_paths.sort()

    wsi.index = 0

    # Non - parallel
    # for wsi_path in wsi.wsi_paths:
    #         wsi.read_wsi_tumor(wsi_path)
    #         wsi.find_roi_n_extract_patches_tumor()

    for g in range(0, len(wsi.wsi_paths), num_threads):
        p = []
        for wsi_path in wsi.wsi_paths[g:g+num_threads]:
            if not os.path.exists(PROCESSED_PATCHES_PATH + wsi_path[wsi_path.find('patient_'):-4]):

                if wsi.read_wsi_tumor(wsi_path):
                    print("Processing (run_on_tumor_data)", wsi_path)
                    pp = multiprocessing.Process(target=wsi.find_roi_n_extract_patches_tumor)
                    p.append(pp)
                    pp.start()
        [pp.join() for pp in p]




wsi = WSI()
run_on_tumor_data()


print("Finished %s"%PATCH_SIZE)

