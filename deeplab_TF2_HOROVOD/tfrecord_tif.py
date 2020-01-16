from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from PIL import Image, ImageStat
from glob import glob as glob
import os
import numpy as np
import cv2
import pdb
import multiprocessing
import tensorflow as tf
import pyvips

TRAIN_PATH = '/lustre4/2/managed_datasets/CAMELYON16/TrainingData/Train_Tumor'
MASK_PATH = '/lustre4/2/managed_datasets/CAMELYON16/TrainingData/Ground_Truth/Mask'
num_threads = 8




# Resize and normalize to [0,1] range
def preprocess_image(image):
  # image = tf.image.decode_jpeg(image, channels=3)
  image = image / 255
  image = 2*image - 1
  return image



def make_tfrecord(wsi,mask,i,wsi_path):

    wsi_name = wsi_path[wsi_path.rfind('/')+1:]
    print("Building TF-Record for ", wsi_name)

    # Convert the images and labels to a Tensorflow 'dataset' object (ds), and normalize to [0,1] range
    train_paths_ds = tf.data.Dataset.from_tensor_slices(wsi)
    train_label_ds = tf.data.Dataset.from_tensor_slices(mask)
    train_image_ds = train_paths_ds.map(preprocess_image)
    train_label_ds = train_label_ds.map(preprocess_image)


    # To optimize data input/output, serialize and save as TFRecords
    train_image_ds = train_image_ds.map(tf.io.serialize_tensor)
    train_label_ds = train_label_ds.map(tf.io.serialize_tensor)

    tfrec_train = tf.data.experimental.TFRecordWriter(wsi_name + '_' + str(i) + '.tfrec')
    tfrec_train.write(train_image_ds)

    tfrec_train_label = tf.data.experimental.TFRecordWriter(wsi_name + '-mask-' + str(i) + '.tfrec')
    tfrec_train_label.write(train_label_ds)


    print("Finished writing TF-Record for ", wsi_name)

    return


def get_image_contours_tumor(cont_img, rgb_image):
    contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours_rgb_image_array = np.array(rgb_image)

    line_color = (255, 0, 0)  # blue color code
    cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 3)
    # cv2.drawContours(mask_image, contours_mask, -1, line_color, 3)
    return contours_rgb_image_array, bounding_boxes


def find_roi_n_extract_patches_tumor(wsi_rgb_image,mask_image):
    hsv = cv2.cvtColor(wsi_rgb_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # (50, 50)
    close_kernel = np.ones((50, 50), dtype=np.uint8)
    image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
    # (30, 30)
    open_kernel = np.ones((30, 30), dtype=np.uint8)
    image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
    contour_rgb, bounding_boxes = get_image_contours_tumor(np.array(image_open), wsi_rgb_image)
    # Image.fromarray(np.array(contour_rgb)).show()
    # pdb.set_trace()
    # contour_rgb = cv2.resize(contour_rgb, (0, 0), fx=0.40, fy=0.40)
    # cv2.imshow('contour_rgb', np.array(contour_rgb))
    # wsi_rgb_image.close()
    return contour_rgb, bounding_boxes


def read_wsi(wsi_path, mask_path):
    """
        # =====================================================================================
        # read WSI image and resize
        # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
        # ======================================================================================
    """
    try:
        cur_wsi_path = wsi_path
        cur_mask_path = mask_path
        wsi_image = OpenSlide(wsi_path)
        mask_image = OpenSlide(mask_path)
        pdb.set_trace()
        rgb_image_pil = wsi_image.read_region((0, 0), 7, wsi_image.level_dimensions[7])
        mask_image_pil = mask_image.read_region((0, 0), 7, wsi_image.level_dimensions[7])

        wsi_rgb_image = np.array(rgb_image_pil)
        mask_image = np.array(mask_image_pil)

        wsi_vips  = pyvips.Image.new_from_file(cur_wsi_path)
        mask_vips = pyvips.Image.new_from_file(cur_mask_path)
        contour_rgb, bounding_boxes = find_roi_n_extract_patches_tumor(wsi_rgb_image,mask_image)


        mag_factor = pow(2, 7)

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))
        for i, bounding_box in enumerate(bounding_boxes):
            # sometimes the bounding boxes annotate a very small area not in the ROI
            if (bounding_box[2] * bounding_box[3]) < 2500:
                continue
            b_x_start = int(bounding_box[0]) * mag_factor
            b_y_start = int(bounding_box[1]) * mag_factor
            b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
            b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor

            bb_vips      = wsi_vips.crop(b_x_start, b_y_start, b_x_end - b_x_start, b_y_end - b_y_start)
            bb_mask_vips = mask_vips.crop(b_x_start, b_y_start, b_x_end - b_x_start, b_y_end - b_y_start)
            print("Loading array of bb %s..."%i)
            bb_arr       = np.ndarray(buffer=bb_vips.write_to_memory(), dtype=np.uint8, shape=[bb_vips.height, bb_vips.width, bb_vips.bands])
            bb_mask_arr  = np.ndarray(buffer=bb_mask_vips.write_to_memory(), dtype=np.uint8, shape=[bb_mask_vips.height, bb_mask_vips.width, bb_mask_vips.bands])

            make_tfrecord(bb_arr, bb_mask_arr, i, wsi_path)

    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return False

    return True

if __name__ == '__main__':

    image_list = [x for x in sorted(glob(TRAIN_PATH + '/*', recursive=True)) if 'Mask' not in x]

    mask_list  = [x for x in sorted(glob(MASK_PATH  + '/*', recursive=True)) if 'Mask' in x]
    train_wsi = list(zip(image_list,mask_list))


    for wsi_path, mask_path in train_wsi:
        read_wsi(wsi_path,mask_path)

    # for g in range(0, len(train_wsi), num_threads):
    #     p = []
    #     for wsi_path, mask_path in train_wsi[g:g + num_threads]:
    #         print("Processing (run_on_tumor_data)", wsi_path)
    #         pp = multiprocessing.Process(target=read_wsi, args=(wsi_path,mask_path))
    #         p.append(pp)
    #         pp.start()
    #     [pp.join() for pp in p]



