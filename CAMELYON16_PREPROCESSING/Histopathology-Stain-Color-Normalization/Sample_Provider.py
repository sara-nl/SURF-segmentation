import numpy as np
#import scipy.misc as misc
# import imageio
from scipy.ndimage import rotate
from ops import find_files
from PIL import Image
import os
import pdb
import cv2
from openslide import OpenSlide, OpenSlideUnsupportedFormatError

# Zoom level for RGB Image from OpenSlide
LEVEL_USED = 7

class SampleProvider(object):
  
  def __init__(self, name, data_dir, fileformat, image_options, is_train):
    self.name = name
    self.is_train = is_train
    self.path = data_dir
    self.fileformat = fileformat
    self.reset_batch_offset()
    self.files = self._create_image_lists()
    self.image_options = image_options
    self._read_images()

    
  def _create_image_lists(self):
    if not os.path.exists(self.path):    
        print("Image directory '" + self.path + "' not found.")
        return None
    
    file_list = list()

    for filename in find_files(self.path, '*.' + self.fileformat):
        file_list.append(filename)
    print ('No. of files: %d' % (len(file_list)))
    return file_list

  def _read_images(self):
    self.__channels = True
    # pdb.set_trace()
    #self.images_org = np.array([misc.imread(filename) for filename in self.files])
    #self.images_org = np.array([imageio.imread(filename) for filename in self.files])

    self.images_org = np.array([OpenSlide(filename) for filename in self.files[:1]])

    self.masks_org = []
    for filename in self.files[:1]:
        filename = filename.replace('Train_Tumor', 'Ground_Truth/Mask')
        filename = filename.replace('.tif', '_Mask.tif')
        self.masks_org.append(OpenSlide(filename))

  @staticmethod
  def _get_image_contours(cont_img, rgb_image):
      contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      bounding_boxes = [cv2.boundingRect(c) for c in contours]
      contours_rgb_image_array = np.array(rgb_image)

      line_color = (255, 0, 0)  # blue color code
      cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 3)
      # cv2.drawContours(mask_image, contours_mask, -1, line_color, 3)
      return contours_rgb_image_array, bounding_boxes

  def _transform(self, images_org, masks_org):
    PATCH_SIZE = int(self.image_options["resize_size"])
    self.rgb_image_pil = images_org.read_region((0, 0), LEVEL_USED,
                                                          images_org.level_dimensions[LEVEL_USED])
    self.rgb_image = np.array(self.rgb_image_pil)

    hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
    if self.path.find('tumor') > -1:
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        # (50, 50)
        close_kernel = np.ones((50, 50), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        # (30, 30)
        open_kernel = np.ones((30, 30), dtype=np.uint8)
    else:
        # [20, 20, 20]
        lower_red = np.array([20, 50, 20])
        # [255, 255, 255]
        upper_red = np.array([200, 150, 200])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        # (50, 50)
        close_kernel = np.ones((25, 25), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        # (30, 30)
        open_kernel = np.ones((30, 30), dtype=np.uint8)





    image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
    contour_rgb, bounding_boxes = self._get_image_contours(np.array(image_open), self.rgb_image)

    # contour_rgb = cv2.resize(contour_rgb, (0, 0), fx=0.40, fy=0.40)
    # cv2.imshow('contour_rgb', np.array(contour_rgb))
    self.rgb_image_pil.close()

    if self.image_options["crop"]:
        mag_factor = pow(2, LEVEL_USED)

    print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

    bounding_box = bounding_boxes[0]
    b_x_start = int(bounding_box[0]) * mag_factor
    b_y_start = int(bounding_box[1]) * mag_factor
    b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
    b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor
    X = np.random.random_integers(b_x_start, high=b_x_end, size=500)
    Y = np.random.random_integers(b_y_start, high=b_y_end, size=500)
    # X = np.arange(b_x_start, b_x_end-256, 5)
    # Y = np.arange(b_y_start, b_y_end-256, 5)

    for x, y  in zip(X, Y):
        patch = images_org.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
        mask = masks_org.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
        mask_gt = np.array(mask)
        mask_gt = cv2.cvtColor(mask_gt[:,:,:3], cv2.COLOR_BGR2GRAY)
        # img = Image.fromarray(mask_gt)
        # img.show()
        patch_array = np.array(patch)
        # img = Image.fromarray(patch_array)
        # img.show()
        white_pixel_cnt_gt = cv2.countNonZero(mask_gt)

        if white_pixel_cnt_gt == 0:  # mask_gt does not contain tumor area
            patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_BGR2HSV)
            # img = Image.fromarray(patch_hsv)
            # img.show()
            lower_red = np.array([20, 20, 20])
            upper_red = np.array([200, 200, 200])
            mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
            # img = Image.fromarray(mask_patch)
            # img.show()
            white_pixel_cnt = cv2.countNonZero(mask_patch)

        if white_pixel_cnt < ((PATCH_SIZE * PATCH_SIZE) * 0.50):
            # print(white_pixel_cnt)
            pass
        else:
            break


    image = np.array(patch_array)
    if self.image_options["resize"]:
        resize_size = int(self.image_options["resize_size"])
        image = image.imresize((resize_size, resize_size))

    if self.image_options["flip"]:
        if (np.random.rand() < .5):
            image = image[::-1, ...]

        if (np.random.rand() < .5):
            image = image[:, ::-1, ...]

    if self.image_options["rotate_stepwise"]:
        if (np.random.rand() > .25):  # skip "0" angle rotation
            angle = int(np.random.permutation([1, 2, 3])[0] * 90)
            image = rotate(image, angle, reshape=False)



    return image

  def get_records(self):
    return self.files, self.annotations
        
  def get_records_info(self):
      return self.files
        
  def reset_batch_offset(self, offset=0):
      self.batch_offset = offset
      self.epochs_completed = 0

  def DrawSample(self, batch_size):
    start = self.batch_offset
    self.batch_offset += batch_size
    if self.batch_offset > len(self.images_org):    #  self.images_org.shape[0]:
        
        if not self.is_train:
            image = []
            return image
            
        # Finished epoch
        self.epochs_completed += 1
        print(">> Epochs completed: #" + str(self.epochs_completed))
        # Shuffle the data
        perm = np.arange(len(self.images_org),dtype=np.int)        #self.images_org.shape[0], dtype=np.int)
        np.random.shuffle(perm)
        
        self.images_org = self.images_org[perm]
        self.files = [self.files[k] for k in perm] 
        
        # Start next epoch
        start = 0
        self.batch_offset = batch_size

    end = self.batch_offset


    image = [self._transform(self.images_org[k],self.masks_org[k]) for k in range(start,end)]



    curfilename = [self.files[k] for k in range(start,end)]
    
    return np.asarray(image), curfilename
    

