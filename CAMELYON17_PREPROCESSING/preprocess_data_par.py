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


parser = argparse.ArgumentParser(description='Preprocessing CAMELYON17',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--patch_size', type=int, default=1024,
                    help='Patch size to generate')

parser.add_argument('--num_threads', type=int, default=4,
                    help='Number of threads used for generating patches, parallelize bounding boxes per WSI')

parser.add_argument('--save_neg', type=bool, default=False,
                    help='Whether the negative patches should be written to disk')

parser.add_argument('--save_png', type=bool, default=False,
                    help='Whether the png images of bounding boxes and saved patches should be written to disk')

parser.add_argument('--proc_center', default=1, type=int,
                    help='Center for preprocessing')

opts = parser.parse_args()

parser.add_argument('--train_tumor_wsi_path', type=str,
                                              help='Folder of where the training data is located',
                                              default=f'/nfs/managed_datasets/CAMELYON17/training/center_{opts.proc_center}')

parser.add_argument('--train_tumor_mask_path',type=str,
                                              help='Folder of where the training data is located',
                                              default='/nfs/managed_datasets/CAMELYON17/training')

parser.add_argument('--save_tumor_negative_path',type=str,
                                                 help='Folder of where the negatives patches are saved.',
                                                 default=f'pro_patch_tumor_negative_{opts.patch_size}/')

parser.add_argument('--save_positive_path',type=str,
                                           help='Folder of where the positive patches are saved.',
                                           default=f'pro_patch_positive_{opts.patch_size}/')

opts = parser.parse_args()

tf_coord = tf.train.Coordinator()


num_threads = opts.num_threads
PATCH_SIZE  = opts.patch_size
id          = opts.proc_center

# modify below directory entries as per your local file system
TRAIN_TUMOR_WSI_PATH = opts.train_tumor_wsi_path
TRAIN_TUMOR_MASK_PATH = opts.train_tumor_mask_path



# modify below directory entries as per your local save directory
PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH = opts.save_tumor_negative_path
if not os.path.exists(PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH):
    os.makedirs(PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH)

PROCESSED_PATCHES_POSITIVE_PATH = opts.save_positive_path
if not os.path.exists(PROCESSED_PATCHES_POSITIVE_PATH):
    os.makedirs(PROCESSED_PATCHES_POSITIVE_PATH)


print("Processing Patch Size ", PATCH_SIZE," of center ", id)
PATCH_NORMAL_PREFIX = f'normal_center_{id}_'
PATCH_TUMOR_PREFIX = f'tumor_center_{id}_'



class WSI(object):
    """
        # ================================
        # Class to annotate WSIs with ROIs
        # Class to annotate WSIs with ROIs
        # ================================

    """
    index = 0
    # 1 = 40x
    downsample = 1
    # 0 = 40x
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

            tumoridx = 0
            ntumoridx = 0
            patchidx = 0


            # for x, y in zip(X, Y):
            #     patch = self.wsi_image.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
            #     mask = self.mask_image.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
            for y_left in range(b_y_start, b_y_end, PATCH_SIZE):

                for x_left in range(b_x_start, b_x_end, PATCH_SIZE):

                    patch = self.wsi_image.read_region((x_left, y_left), self.level_output, (PATCH_SIZE, PATCH_SIZE))
                    mask = self.mask_image.read_region((x_left, y_left), 0, (PATCH_SIZE, PATCH_SIZE))


                    _std = ImageStat.Stat(patch).stddev
                    # thresholding stddev for patch extraction
                    patchidx += 1


                    orx = int(x_left / mag_factor)
                    ory = int(y_left / mag_factor)
                    orps = int(PATCH_SIZE / mag_factor)

                    d1.rectangle([(orx,ory),(orx + orps,ory+orps)],outline='green',width=1)

                    mask_gt = np.array(mask)
                    mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
                    patch_array = np.array(patch)


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



                    white_pixel_cnt_gt = cv2.countNonZero(mask_gt)
                    if white_pixel_cnt_gt == 0:  # mask_gt does not contain tumor area
                        patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_BGR2HSV)
                        lower_red = np.array([0, 0, 0])
                        upper_red = np.array([200, 200, 220])
                        mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
                        white_pixel_cnt = cv2.countNonZero(mask_patch)

                        if white_pixel_cnt > ((PATCH_SIZE * PATCH_SIZE) * 0.50):
                            if opts.save_neg:
                                patch.save(PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH + PATCH_NORMAL_PREFIX +str(PATCH_SIZE)+'_'+
                                           str(self.negative_patch_index)+ '.png')
                                mask.save(PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH + 'mask_' + PATCH_NORMAL_PREFIX+str(PATCH_SIZE)+'_'+ str(self.negative_patch_index)+ '.png')
                            self.negative_patch_index += 1
                            ntumoridx+=1

                    else:  # mask_gt contains tumor area
                        if white_pixel_cnt_gt >= ((PATCH_SIZE * PATCH_SIZE) * 0.01):
                            patch.save(PROCESSED_PATCHES_POSITIVE_PATH + PATCH_TUMOR_PREFIX +str(PATCH_SIZE)+'_'+
                                       str(self.positive_patch_index) + '.png')
                            mask.save(PROCESSED_PATCHES_POSITIVE_PATH + 'mask_' + PATCH_TUMOR_PREFIX +str(PATCH_SIZE)+'_'+
                                       str(self.positive_patch_index)+ '.png')

                            self.positive_patch_index += 1
                            tumoridx+=1
                            DIR = PROCESSED_PATCHES_POSITIVE_PATH
                            file_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
                            if file_count >= 1000:
                                print("Finished 500 positives")
                                sys.exit(0)


                    patch.close()
                    mask.close()
            print("Processed patches in bounding box %s :" % i, "%s" % patchidx, " positive: %s " % tumoridx,
                  " negative: %s" % ntumoridx)

        if opts.save_png:
            rgb_image_pil.save('bb_%s.png'%(self.cur_wsi_path[self.cur_wsi_path.rfind('/')+1:]))
            rgb_image_pilpro.save('bbpro_%s.png'%(self.cur_wsi_path[self.cur_wsi_path.rfind('/')+1:]))

    def read_wsi_mask(self, wsi_path, mask_path):
        try:
            self.cur_wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)
            self.mask_image = OpenSlide(mask_path)

            self.level_used = min(self.def_level, self.wsi_image.level_count - 1, self.mask_image.level_count - 1)

            self.mask_pil = self.mask_image.read_region((0, 0), self.level_used,
                                                            self.mask_image.level_dimensions[self.level_used])
            self.mask = np.array(self.mask_pil)

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return False

        return True



    @staticmethod
    def convert_contour_coordinate_resolution(l_contours, downsample):
        """
            convert  contours coordinate to downsample resolution
            input:
            l_contours : list of contours coordinate(x,y) in level 0 resolution
            downsample : disired resolution
            return coverted contour list
        """
        cvted_l_contours = []

        for contour in l_contours:
            print('shape of tumor contours: ', contour.shape)
            downsample_coor = contour / downsample
            downsample_coor = (downsample_coor).astype(int)
            downsample_coor = np.unique(downsample_coor, axis=0)
            cvted_l_contours.append(downsample_coor)

        return cvted_l_contours

    def make_list_of_contour_from_xml(self,fn_xml,downsample):

        """
            make the list of contour from xml(annotation file)
            input:
            fn_xml = file name of xml file
            downsample = desired resolution
            var:
            li_li_point = list of tumors
            li_point = the coordinates([x,y]) of a tumor
            return  list of list (2D array list)
        """

        li_li_point = []
        tree = parse(fn_xml)
        for parent in tree.getiterator():
            for i_1, child1 in enumerate(parent):
                for i_2, child2 in enumerate(child1):
                    for i_3, child3 in enumerate(child2):
                        li_point = []
                        for i_4, child4 in enumerate(child3):
                            x_0 = float(child4.attrib['X'])
                            y_0 = float(child4.attrib['Y'])
                            x_s = x_0 / downsample
                            y_s = y_0 / downsample
                            li_point.append([x_s, y_s])

                        if len(li_point):
                            li_li_point.append(li_point)

        return li_li_point

    @staticmethod
    def convert_list_of_contour_2_opencv_contours(li_li_point):
        """
            conver list of contour(2D list array) to opencv contours
            that list of nparray (not 2-d nparray !)
            input:
            li_li_point = list of contours
            var:
            countours = list of contours
            contour = nparray with x,y coordinate
            return opencv contours
        """

        contours = []

        for li_point in li_li_point:
            li_point_int = [[int(round(point[0])), int(round(point[1]))] for point in li_point]
            contour = np.array(li_point_int, dtype=np.int32)
            contours.append(contour)

        return contours

    def get_opencv_contours_from_xml(self,fn_xml, downsample):

        """"
            get opencv contours( list of nparrays) from xml annotation file
            input:
            fn_xml = xml file name
            downsample = disired downsample
            return list of contours
        """

        li_li_point = self.make_list_of_contour_from_xml(fn_xml, downsample)
        l_contours = self.convert_list_of_contour_2_opencv_contours(li_li_point)

        return l_contours


    def get_mask_from_opencv_contours(self,l_contours, slide, level):
        """
            get binary image map in certain level(resolution)
            input:
            l_contour = list of nparray that contain coordinate(x,y)
            slide = to obtain dimension of mask
            level = desired level
            return tumor mask image (binary image 0-1)
        """

        slid_lev_w, slid_lev_h = slide.level_dimensions[level]
        # mask_image = np.zeros((slid_lev_w, slid_lev_h),dtype = np.int8)
        mask_image = Image.new('1', (slid_lev_w, slid_lev_h))

        print(f"mask_image dimension of {wsi.cur_wsi_path}: {mask_image.size}")
        downsample = slide.level_downsamples[level]
        print(f"downsample: {downsample}")

        # convert coordinate to the level resolution from level=0
        print(f"tumor regions : {len(l_contours)}")
        for i, npary in enumerate(l_contours):

            # down_coordi = npary/float(downsample)
            # down_coordi = (down_coordi).astype(int)
            # print downsample
            # check 1 in the tummor region

            li_xy = npary.flatten()
            d_x, d_y = li_xy[::2], li_xy[1::2]
            contour_coordinates = list(zip(d_x, d_y))
            # mask_image[d_x, d_y] = 255.0
            d = ImageDraw.Draw(mask_image)
            d.polygon(contour_coordinates,fill='white')

        return mask_image

    def read_wsi_tumor(self, wsi_path, mask_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            self.cur_wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)
            ### CAM17 ###
            l_contours      = self.get_opencv_contours_from_xml(mask_path, wsi.downsample)
            self.mask_image = self.get_mask_from_opencv_contours(l_contours,self.wsi_image, self.level_output)
            self.mask_image = ImageSlide(self.mask_image)
            # self.level_used = min(self.def_level, self.wsi_image.level_count - 1, self.mask_image.level_count - 1)
            self.level_used = self.def_level

            self.rgb_image_pil = self.wsi_image.read_region((0, 0), self.level_used, self.wsi_image.level_dimensions[self.level_used])
            self.rgb_image = np.array(self.rgb_image_pil)

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


        
        # Image.fromarray(np.array(contour_rgb)).show()
        # contour_rgb = cv2.resize(contour_rgb, (0, 0), fx=0.40, fy=0.40)
        # cv2.imshow('contour_rgb', np.array(contour_rgb))
        self.rgb_image_pil.close()
        self.extract_patches_tumor(bounding_boxes)
        self.wsi_image.close()
        self.mask_image.close()



    @staticmethod
    def get_image_contours_tumor(cont_img, rgb_image):
        contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        contours_rgb_image_array = np.array(rgb_image)

        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(contours_rgb_image_array, contours, -1, (255,150,150), 1)
        
        # cv2.drawContours(mask_image, contours_mask, -1, line_color, 3)
        return contours_rgb_image_array, bounding_boxes




def run_on_tumor_data():
    wsi.wsi_paths = glob.glob(os.path.join(TRAIN_TUMOR_WSI_PATH, '*.tif'))
    wsi.wsi_paths.sort()
    # Get annotation xml files to construct mask
    for slide in wsi.wsi_paths:
        # get separated tif part
        tif = slide[slide.rfind('/'):]
        xml = tif.replace('.tif', '.xml')
        # check if tumor lesion
        if os.path.isfile(TRAIN_TUMOR_MASK_PATH + xml):
            wsi.mask_paths.append(TRAIN_TUMOR_MASK_PATH + xml)


    wsi.mask_paths.sort()

    wsi.index = 0

    # get only wsi_paths that match the tumor xml's
    wsi.wsi_paths = [el for el in wsi.wsi_paths if el.replace(TRAIN_TUMOR_WSI_PATH[TRAIN_TUMOR_WSI_PATH.rfind('/'):],'').replace('.tif','.xml') in wsi.mask_paths]
    assert len(wsi.wsi_paths) == len(wsi.mask_paths), "Number of images is not equal to number of annotations"
    lst_done = [x[5:-4] for x in glob.glob(os.path.join('./', 'bb_*.png'))]


    if num_threads == 1:

        # Non - parallel
        for wsi_path, mask_path in zip(wsi.wsi_paths, wsi.mask_paths):
            if wsi_path[wsi_path.rfind('/') + 1:] not in lst_done:
                wsi.read_wsi_tumor(wsi_path, mask_path)
                wsi.find_roi_n_extract_patches_tumor()

    elif num_threads > 1:
        for g in range(0, len(wsi.wsi_paths), num_threads):
            p = []
            for wsi_path, mask_path in zip(wsi.wsi_paths[g:g+num_threads], wsi.mask_paths[g:g+num_threads]):
                if wsi_path[wsi_path.rfind('/') + 1:] not in lst_done:
                    if wsi.read_wsi_tumor(wsi_path, mask_path):
                        print("Processing (run_on_tumor_data)", wsi_path)
                        pp = multiprocessing.Process(target=wsi.find_roi_n_extract_patches_tumor)
                        p.append(pp)
                        pp.start()
            [pp.join() for pp in p]

    else:
        raise SyntaxError("Specify --num_threads in args")



if __name__ == "__main__":
    wsi = WSI()
    run_on_tumor_data()
    print(f"Finised {PATCH_SIZE}")