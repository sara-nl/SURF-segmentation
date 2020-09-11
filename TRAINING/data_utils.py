import numpy as np
from glob import glob
import os
from sklearn.utils import shuffle
from PIL import Image, ImageStat
import tensorflow as tf
import horovod.tensorflow as hvd
import pdb
import pyvips
import cv2
import sys
import random
import time
from openslide import OpenSlide, ImageSlide, OpenSlideUnsupportedFormatError
from utils import rank00
sys.path.insert(0,'/home/rubenh/SURF-deeplab/TRAINING/xml-pathology')
from xmlpathology.batchgenerator.utils import create_data_source
from xmlpathology.batchgenerator.generators import BatchGenerator
from xmlpathology.batchgenerator.core.samplers import LabelSamplerLoader, SamplerLoader
from xmlpathology.batchgenerator.core.samplers import SegmentationLabelSampler, Sampler
from xmlpathology.batchgenerator.callbacks import OneHotEncoding, FitData


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


class PreProcess():
    def __init__(self):
        pass
    
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
    
    
class RadSampler(PreProcess):
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
        
        
        datasource_train = create_data_source(data_folder=opts.slide_path,
                                            annotations_path=opts.label_path,
                                            images_extension='.'+self.slide_format,
                                            annotations_extension='.'+self.label_format,
                                            mode='training')

        if rank00(): print(f"Found {len(datasource_train['training'])} training images")        
        datasource_validation = create_data_source(data_folder=opts.valid_slide_path,
                                                   annotations_path=opts.valid_label_path,
                                                   images_extension='.'+self.valid_slide_format,
                                                   annotations_extension='.'+self.valid_label_format,
                                                   mode='validation')
        if rank00(): print(f"Found {len(datasource_validation['validation'])} validation images")
          

        # initialize batchgenerator
        if rank00(): print("Starting Training Batch Generator...")
        self.batchgen_train = BatchGenerator(data_sources=datasource_train,
                                        label_map=self.label_map,
                                        batch_size=self.batch_size,
                                        cpus=self.cpus,
                                        sampler_loader=SamplerLoader(class_=Sampler, input_shapes=[[self.patch_size,self.patch_size,3]],
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
                                                                          input_shapes=[[self.patch_size,self.patch_size,3]],
                                                                          spacings=self.resolution),
                                             label_sampler_loader=LabelSamplerLoader(class_=SegmentationLabelSampler),
                                             log_path=opts.log_dir)
        self.batchgen_validation.start()

        
        
    def get_next(self,train=True):
        
        # Every new iteration, new sample
        if train:
            # Ugly, but rescaling in place
            im = (self.batchgen_train.batch('training')['x_batch']*255).astype('float32')
            msk = self.batchgen_train.batch('training')['y_batch'][...,None].astype('float32')
            dataset = tf.data.Dataset.from_tensor_slices((im,msk))
            dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                map_func=lambda im, msk: RadSampler._load(im,msk,augment=False),
                batch_size=self.batch_size,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                drop_remainder=True))
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            # Ugly, but rescaling in place
            im = (self.batchgen_validation.batch('validation')['x_batch']*255).astype('float32')
            msk = self.batchgen_validation.batch('validation')['y_batch'][...,None].astype('float32')
            dataset = tf.data.Dataset.from_tensor_slices((im,msk))
            dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                map_func=lambda im, msk: RadSampler._load(im,msk,augment=False),
                batch_size=self.batch_size,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                drop_remainder=True))
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
            
                    


class SurfSampler(PreProcess):
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
    
    
    !! Label and WSI directory must adhere to same sorting order
    
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
    def __init__(self, opts):
        super().__init__()
        self.train_paths = shuffle(list(zip(sorted(glob(os.path.join(opts.slide_path,f'*.{opts.slide_format}'))),
                                            sorted(glob(os.path.join(opts.label_path,f'*.{opts.label_format}'))))))
        
        if rank00(): print(f"Found {len(self.train_paths)} images")
        if opts.valid_slide_path:
            self.valid_paths = shuffle(list(zip(sorted(glob(os.path.join(opts.valid_slide_path,f'*.{opts.slide_format}'))),
                                                sorted(glob(os.path.join(opts.valid_label_path,f'*.{opts.label_format}'))))))
        else:
            val_split = int(len(self.train_paths) * opts.val_split)
            self.valid_paths = self.train_paths[val_split:]
            self.train_paths = self.train_paths[:val_split]
            
        
        self.batch_size = opts.batch_size
        self.contours_train = []
        self.contours_valid = []
        self.contours_tumor = []
        self.level_used = opts.bb_downsample
        self.mag_factor = pow(2, self.level_used)
        self.patch_size = opts.img_size
        self.tumor_ratio = opts.batch_tumor_ratio
        self.log_image_path = opts.log_dir
        self.slide_format = opts.slide_format
        
        
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
        _offset=0
        for i, contour in enumerate(contours):
            # sometimes the bounding boxes annotate a very small area not in the ROI
            if contour.shape[0] < 10:
                if rank00(): print(f"Deleted too small contour from {self.cur_wsi_path}")
                del contours[i]
                _offset+=1
                i=i-_offset
        # contours_rgb_image_array = np.array(self.rgb_image)
        # line_color = (255, 150, 150)  
        # cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 1)
        # Image.fromarray(contours_rgb_image_array[...,:3]).save('test.png')

        # self.rgb_image_pil.close()
        # self.wsi.close()
        # self.mask.close()
        
        return contours
    
    
            
    def get_next(self,train=True):
        
        # Every new iteration, new sample
        if train:
            while not self.contours_train or not self.contours_tumor:
    
                self.cur_wsi_path = random.choice(self.train_paths)
                if rank00(): print(f"Opening {self.cur_wsi_path}...")
                
                self.wsi  = OpenSlide(self.cur_wsi_path[0])
                self.mask = OpenSlide(self.cur_wsi_path[1])
                
                self.rgb_image_pil = self.wsi.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                self.mask_pil = self.mask.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                self.rgb_image = np.array(self.rgb_image_pil)
                self.mask_image = np.array(self.mask_pil)
                self.bounding_boxes_train = self.get_bb()
                self.contours_train = self.get_bb()
                self.contours = self.contours_train
                
                # Get contours of tumor
                contours, _ = cv2.findContours(self.mask_image[...,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self.contours_tumor = contours
            
        
        else:
            while not self.contours_valid or not self.contours_tumor:
    
                self.cur_wsi_path = random.choice(self.valid_paths)
                if rank00(): print(f"Opening {self.cur_wsi_path}...")
                
                self.wsi  = OpenSlide(self.cur_wsi_path[0])
                self.mask = OpenSlide(self.cur_wsi_path[1])
                
                
                self.rgb_image_pil = self.wsi.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                self.mask_pil = self.mask.read_region((0, 0), self.level_used, self.wsi.level_dimensions[self.level_used])
                self.rgb_image = np.array(self.rgb_image_pil)
                self.mask_image = np.array(self.mask_pil)
                self.bounding_boxes_valid = self.get_bb()
            
                self.contours_valid = self.get_bb()
                self.contours = self.contours_valid
                
                # Get contours of tumor
                contours, _ = cv2.findContours(self.mask_image[...,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self.contours_tumor = contours

        image = pyvips.Image.new_from_file(self.cur_wsi_path[0])
        mask_image  = pyvips.Image.new_from_file(self.cur_wsi_path[1])
        img_reg = pyvips.Region.new(image)
        mask_reg = pyvips.Region.new(mask_image)
        
        numpy_batch_patch = []
        numpy_batch_mask  = []
        if os.path.isfile(os.path.join(self.log_image_path,self.cur_wsi_path[0].split('/')[-1].replace(self.slide_format,'png'))):
            try:
                save_image = np.array(Image.open(os.path.join(self.log_image_path,self.cur_wsi_path[0].split('/')[-1].replace(self.slide_format,'png'))))
            except:
                save_image = self.rgb_image.copy() * np.repeat((self.mask_image + 1)[...,0][...,np.newaxis],4,axis=-1)
        else:
        # copy image and mark tumor in black
            save_image = self.rgb_image.copy() * np.repeat((self.mask_image + 1)[...,0][...,np.newaxis],4,axis=-1)
        
        for i in range(int(self.batch_size * (1 - self.tumor_ratio))):
            bc = random.choice(self.contours)
            msk = np.zeros(self.rgb_image.shape,np.uint8)
            cv2.drawContours(msk,[bc],-1,(255),-1)
            pixelpoints = np.transpose(np.nonzero(msk))
            
            b_x_start = bc[...,0].min() * self.mag_factor
            b_y_start = bc[...,1].min() * self.mag_factor
            b_x_end = bc[...,0].max() * self.mag_factor
            b_y_end = bc[...,1].max() * self.mag_factor
            h = b_y_end - b_y_start
            w = b_x_end - b_x_start
        
            patch = []
            k=0
            while not len(patch):
                x_topleft = random.choice(pixelpoints)[1]* self.mag_factor
                y_topleft = random.choice(pixelpoints)[0]* self.mag_factor
                t1 = time.time()
                # if trying to fetch outside of image, retry
                try:
                    patch = img_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
                    patch = np.ndarray((self.patch_size,self.patch_size,image.get('bands')),buffer=patch, dtype=np.uint8)[...,:3]
                    _std = ImageStat.Stat(Image.fromarray(patch)).stddev
                    k+=1
                    
                    # discard based on stddev
                    if k < 5: 
                        if (sum(_std[:3]) / len(_std[:3])) < 15:
                            print("Discard based on stddev")
                            patch = []
                except Exception as e:
                    print("Exception in extracting patch: ", e)
                    patch = []

            if rank00(): print(f"Sample {self.patch_size} x {self.patch_size} from contours = {h}" + f" by {w} in {time.time() -t1} seconds")
            numpy_batch_patch.append(patch)
            mask  = mask_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
            numpy_batch_mask.append(np.ndarray((self.patch_size,self.patch_size,mask_image.get('bands')),buffer=mask, dtype=np.uint8))
            
            try:
                # Draw the rectangles of sampled images on downsampled rgb
                save_image = cv2.drawContours(save_image, self.contours, -1, (0,255,0), 1)
                save_image = cv2.rectangle(save_image, (int(x_topleft // self.mag_factor) , int(y_topleft // self.mag_factor)),
                                                       (int((x_topleft + self.patch_size) // self.mag_factor), int((y_topleft + self.patch_size) // self.mag_factor)),
                                                       (0,255,0), 2)
            except:
                pass
        
        for i in range(int(self.batch_size * (self.tumor_ratio))):
            bc = random.choice(self.contours_tumor)
            msk = np.zeros(self.rgb_image.shape,np.uint8)
            cv2.drawContours(msk,[bc],-1,(255),-1)
            pixelpoints = np.transpose(np.nonzero(msk))
            
            b_x_start = bc[...,0].min() * self.mag_factor
            b_y_start = bc[...,1].min() * self.mag_factor
            b_x_end = bc[...,0].max() * self.mag_factor
            b_y_end = bc[...,1].max() * self.mag_factor
            h = b_y_end - b_y_start
            w = b_x_end - b_x_start
        
            patch = []
            
            k=0
            while not len(patch):
                x_topleft = random.choice(pixelpoints)[1]* self.mag_factor
                y_topleft = random.choice(pixelpoints)[0]* self.mag_factor
                t1 = time.time()
                # if trying to fetch outside of image, retry
                try:
                    patch = img_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
                    patch = np.ndarray((self.patch_size,self.patch_size,image.get('bands')),buffer=patch, dtype=np.uint8)[...,:3]
                    _std = ImageStat.Stat(Image.fromarray(patch)).stddev
                    
                    k+=1
                    # discard based on stddev
                    if k < 5: 
                        if (sum(_std[:3]) / len(_std[:3])) < 15:
                            print("Discard based on stddev")
                            patch = []
                    
                except Exception as e:
                    print("Exception in extracting patch: ",e)
                    patch = []

            if rank00(): print(f"Sample {self.patch_size} x {self.patch_size} from contours = {h}" + f" by {w} in {time.time() -t1} seconds")
            numpy_batch_patch.append(patch)
            mask  = mask_reg.fetch(x_topleft, y_topleft, self.patch_size, self.patch_size)
            numpy_batch_mask.append(np.ndarray((self.patch_size,self.patch_size,mask_image.get('bands')),buffer=mask, dtype=np.uint8))
        
            try:
                # Draw the rectangles of sampled images on downsampled rgb
                save_image = cv2.drawContours(save_image, self.contours, -1, (0,255,0), 1)
                save_image = cv2.rectangle(save_image, (int(x_topleft // self.mag_factor) , int(y_topleft // self.mag_factor)),
                                                       (int((x_topleft + self.patch_size) // self.mag_factor), int((y_topleft + self.patch_size) // self.mag_factor)),
                                                       (0,255,0), 2)
            except:
                pass

        # Image.fromarray(save_image[...,:3]).save(os.path.join(self.log_image_path,self.cur_wsi_path[0].split('/')[-1].replace(self.slide_format,'png')))
        dataset = tf.data.Dataset.from_tensor_slices((np.array(numpy_batch_patch),np.array(numpy_batch_mask)))
        dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=lambda im, msk: SurfSampler._load(im,msk,augment=False),
            batch_size=self.batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            drop_remainder=True))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.contours_train = []
        self.contours_valid = []
        self.contours_tumor = []
        
        return dataset
        self.contours_tumor = []
        return dataset







