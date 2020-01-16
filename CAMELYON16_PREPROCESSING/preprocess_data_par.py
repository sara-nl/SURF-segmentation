from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from PIL import Image, ImageStat, ImageDraw
import glob
import os
import numpy as np
import cv2
import pdb
import multiprocessing
import tensorflow as tf
import argparse


parser = argparse.ArgumentParser(description='Preprocessing CAMELYON16',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--patch_size', type=int, default=1024,
                    help='Patch size to generate')
parser.add_argument('--num_threads', type=int, default=4,
                    help='Number of threads used for generating patches, parallelize bounding boxes per WSI')

parser.add_argument('--save_neg', type=bool, default=False,
                    help='Whether the negative patches should be written to disk')

parser.add_argument('--save_png', type=bool, default=False,
                    help='Whether the png images of bounding boxes and saved patches should be written to disk')

opts = parser.parse_args()

tf_coord = tf.train.Coordinator()


num_threads = opts.num_threads
PATCH_SIZE  = opts.patch_size



# modify below directory entries as per your local file system
TRAIN_TUMOR_WSI_PATH = '/nfs/managed_datasets/CAMELYON16/TrainingData/Train_Tumor'
TRAIN_TUMOR_MASK_PATH = '/nfs/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/Mask'

# modify below directory entries as per your local save directory
PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH ='pro_patch_tumor_negative_%s/'%PATCH_SIZE
PROCESSED_PATCHES_POSITIVE_PATH ='pro_patch_positive_%s/'%PATCH_SIZE

print("Processing Patch Size ", PATCH_SIZE)
PATCH_NORMAL_PREFIX = 'normal_'
PATCH_TUMOR_PREFIX = 'tumor_'



class WSI(object):
    """
        # ================================
        # Class to annotate WSIs with ROIs
        # ================================

    """
    index = 0
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
        mag_factor = pow(2, self.level_used)

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        rgb_image_pil = Image.fromarray(self.rgb_image)
        d1 = ImageDraw.Draw(rgb_image_pil)

        rgb_image_pilpro = Image.fromarray(self.rgb_image)
        d2 = ImageDraw.Draw(rgb_image_pilpro)

        print("Processing WSI %s" % self.cur_wsi_path)

        for i, bounding_box in enumerate(bounding_boxes):
            # sometimes the bounding boxes annotate a very small area not in the ROI
            if (bounding_box[2] * bounding_box[3]) < 2500:
                print("Skipped too small Bounding Box %s" % self.cur_wsi_path)
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


                    patch = self.wsi_image.read_region((x_left, y_left), 0, (PATCH_SIZE, PATCH_SIZE))
                    mask = self.mask_image.read_region((x_left, y_left), 0, (PATCH_SIZE, PATCH_SIZE))
                    _std = ImageStat.Stat(patch).stddev
                    # thresholding stddev for patch extraction
                    patchidx += 1

                    orx = int(x_left / mag_factor)
                    ory = int(y_left / mag_factor)
                    orps = int(PATCH_SIZE / mag_factor)

                    d1.rectangle([(orx, ory), (orx + orps, ory + orps)], outline='green', width=1)

                    mask_gt = np.array(mask)
                    mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
                    patch_array = np.array(patch)

                    # discard based on stddev
                    if (sum(_std[:3]) / len(_std[:3])) < 15:
                        continue

                    # discard based on black pixels
                    gs = cv2.cvtColor(patch_array, cv2.COLOR_BGR2GRAY)
                    black_pixel_cnt_gt = gs.shape[0] * gs.shape[1] - cv2.countNonZero(gs)
                    if black_pixel_cnt_gt > 10:
                        continue

                    d2.rectangle([(orx, ory), (orx + orps, ory + orps)], outline='green', width=1)

                    white_pixel_cnt_gt = cv2.countNonZero(mask_gt)
                    if white_pixel_cnt_gt == 0:  # mask_gt does not contain tumor area
                        patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_BGR2HSV)
                        lower_red = np.array([0, 0, 0])
                        upper_red = np.array([200, 200, 220])
                        mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
                        white_pixel_cnt = cv2.countNonZero(mask_patch)

                        if white_pixel_cnt > ((PATCH_SIZE * PATCH_SIZE) * 0.50):
                            mask = Image.fromarray(mask)
                            if opts.save_neg:
                                patch.save(PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH + PATCH_NORMAL_PREFIX +str(PATCH_SIZE)+'_'+
                                           str(self.negative_patch_index), 'PNG')
                                mask.save(PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH + 'mask_' + PATCH_NORMAL_PREFIX+str(PATCH_SIZE)+'_'+ str(self.negative_patch_index),
                                           'PNG')
                            self.negative_patch_index += 1
                            ntumoridx+=1
                    else:  # mask_gt contains tumor area
                        if white_pixel_cnt_gt >= ((PATCH_SIZE * PATCH_SIZE) * 0.01):
                            patch.save(PROCESSED_PATCHES_POSITIVE_PATH + PATCH_TUMOR_PREFIX +str(PATCH_SIZE)+'_'+
                                       str(self.positive_patch_index), 'PNG')
                            mask.save(PROCESSED_PATCHES_POSITIVE_PATH + 'mask_' + PATCH_TUMOR_PREFIX +str(PATCH_SIZE)+'_'+
                                       str(self.positive_patch_index), 'PNG')

                            self.positive_patch_index += 1
                            tumoridx+=1


                    patch.close()
                    mask.close()

            print("Processed patches in bounding box %s :" % i, "%s" % patchidx, " positive: %s " % tumoridx,
                  " negative: %s" % ntumoridx)

        if opts.save_png:
            rgb_image_pil.save('bb_%s_%s.png'%(PATCH_SIZE, self.cur_wsi_path[self.cur_wsi_path.rfind('/')+1:]))
            rgb_image_pilpro.save('bbpro_%s_%s.png'%(PATCH_SIZE, self.cur_wsi_path[self.cur_wsi_path.rfind('/')+1:]))

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


    def read_wsi_tumor(self, wsi_path, mask_path):
        """
            # =====================================================================================
     i       # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            self.cur_wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)
            self.mask_image = OpenSlide(mask_path)

            self.level_used = min(self.def_level, self.wsi_image.level_count - 1, self.mask_image.level_count - 1)

            self.rgb_image_pil = self.wsi_image.read_region((0, 0), self.level_used,
                                                            self.wsi_image.level_dimensions[self.level_used])
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
        # pdb.set_trace()
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
        cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 3)
        # cv2.drawContours(mask_image, contours_mask, -1, line_color, 3)
        return contours_rgb_image_array, bounding_boxes



def run_on_tumor_data():
    wsi.wsi_paths = glob.glob(os.path.join(TRAIN_TUMOR_WSI_PATH, '*.tif'))
    wsi.wsi_paths.sort()
    wsi.mask_paths = glob.glob(os.path.join(TRAIN_TUMOR_MASK_PATH, '*.tif'))
    wsi.mask_paths.sort()
    

    wsi.index = 0


    # Parallel(n_jobs=8)(delayed(wsi.find_roi_n_extract_patches_tumor)() for wsi_path, mask_path in zip(wsi.wsi_paths, wsi.mask_paths) if wsi.read_wsi_tumor(wsi_path, mask_path))
    assert len(wsi.wsi_paths) == len(wsi.mask_paths), "Not all images have masks"

    lst_done = [x[5:-4] for x in glob.glob(os.path.join('./', 'bb_*.png'))]


    if num_threads == 1:

        # Non - parallel
        for wsi_path, mask_path in zip(wsi.wsi_paths, wsi.mask_paths):
            if '%s_' % PATCH_SIZE + wsi_path[wsi_path.rfind('/') + 1:] not in lst_done:
                if wsi.read_wsi_tumor(wsi_path, mask_path):
                    wsi.find_roi_n_extract_patches_tumor()

    elif num_threads > 1:
        for g in range(0, len(wsi.wsi_paths), num_threads):
            p = []
            for wsi_path, mask_path in zip(wsi.wsi_paths[g:g+num_threads], wsi.mask_paths[g:g+num_threads]):
                print("Testing",wsi_path[wsi_path.rfind('/') + 1:] )
                if '%s_'%PATCH_SIZE + wsi_path[wsi_path.rfind('/') + 1:] not in lst_done:
                    if wsi.read_wsi_tumor(wsi_path, mask_path):
                        print("Processing (run_on_tumor_data)", wsi_path)
                        pp = multiprocessing.Process(target=wsi.find_roi_n_extract_patches_tumor)
                        p.append(pp)
                        pp.start()
                [pp.join() for pp in p]

    else:
        raise SyntaxError("Specify --num_threads in args")



if __name__ == '__main__':
    wsi = WSI()
    run_on_tumor_data()
    print("Finished Patch Size %s"%PATCH_SIZE)

