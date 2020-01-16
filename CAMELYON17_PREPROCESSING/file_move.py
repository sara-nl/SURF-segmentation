import glob
import os
import pdb
from shutil import copyfile
import warnings


"""
NEEDED FILESTRUCTURE:

Processed/
        patch-based-classification_224/
                raw_data/
                        train/
                                label-0
                                label-1
                        validation/
                                label-0
                                label-1
                


Script to move images from  directory to 
.../deeplab/models/research/deeplab/CAMELYON16_PREPROCESSING/Processed/* directory

"""

DIR = '/home/rubenh/projects/deeplab/models/research/deeplab/CAMELYON16_PREPROCESSING/Processed/patch-based-classification_704/raw-data/train/label-0/'
positive_label = 'label-1'
negative_label = 'label-0'
IMG_SIZE = 704
OUT_DIR_PATTERN='patch-based-classification_704_Normalized'
VAL_SPLIT = 0.15

images = []
masks =[]
for image in glob.glob(os.path.join(DIR,'*%s*')%IMG_SIZE):
        if image.find('mask') == -1:
                # if os.path.getsize(image) < 800000:
                #         continue
                images.append(image)

for mask in glob.glob(os.path.join(DIR,'*mask*')):
        # if os.path.getsize(mask.replace('mask_','')) < 800000:
        #         continue
        masks.append(mask)


images = sorted(images)#, key=takeSecond)
masks  = sorted(masks)#, key=takeSecond)

# idx = int(len(images)*(1-VAL_SPLIT))
idx=2000

if len(images) != len(masks):
        warnings.warn("WARNING: Length of images is not equal to length masks")

# Negative images
if DIR.find('%s'%positive_label) == -1:
        print("Begin copying...")
        try:
                ### TRAIN ###
                for i in range(len(images[:idx])):
                        copyfile(images[i], '/home/rubenh/projects/deeplab/models/research/deeplab/CAMELYON16_PREPROCESSING/Processed/%s/raw-data/train/label-0/'%OUT_DIR_PATTERN
                                 + images[i][images[i].rfind('/')+1:])

                        maskname = images[i][:images[i].rfind('/')+1] + 'mask_' + images[i][images[i].rfind('/')+1:]
                        match = [s for s in masks if maskname == s]
                        copyfile(match[0], '/home/rubenh/projects/deeplab/models/research/deeplab/CAMELYON16_PREPROCESSING/Processed/%s/raw-data/train/label-0/'%OUT_DIR_PATTERN + maskname[maskname.rfind('/')+1:])


                print("Finished copying train...")

                ### VALIDATION ###

                # for i in range(idx,len(images)):
                for i in range(idx, 2500):
                        copyfile(images[i], '/home/rubenh/projects/deeplab/models/research/deeplab/CAMELYON16_PREPROCESSING/Processed/%s/raw-data/validation/label-0/'%OUT_DIR_PATTERN
                                 + images[i][images[i].rfind('/')+1:])

                        maskname = images[i][:images[i].rfind('/')+1] + 'mask_' + images[i][images[i].rfind('/')+1:]
                        match = [s for s in masks if maskname == s]

                        copyfile(match[0],  '/home/rubenh/projects/deeplab/models/research/deeplab/CAMELYON16_PREPROCESSING/Processed/%s/raw-data/validation/label-0/'%OUT_DIR_PATTERN
                                 + maskname[maskname.rfind('/')+1:])
        except Exception as e:
                print(e)
                pdb.set_trace()

        print("Finished copying")

# Positive images
elif DIR.find('%s'%negative_label) == -1:

        print("Begin copying...")
        ### TRAIN ###

        for i in range(len(images[:idx])):
                copyfile(images[i],
                          '/home/rubenh/projects/deeplab/models/research/deeplab/CAMELYON16_PREPROCESSING/Processed/%s/raw-data/train/label-1/'%OUT_DIR_PATTERN
                          + images[i][images[i].rfind('/')+1:])

                maskname = images[i][:images[i].rfind('/')+1] + 'mask_' + images[i][images[i].rfind('/')+1:]
                match = [s for s in masks if maskname == s]

                if len(match):
                        copyfile(match[0],
                                  '/home/rubenh/projects/deeplab/models/research/deeplab/CAMELYON16_PREPROCESSING/Processed/%s/raw-data/train/label-1/'%OUT_DIR_PATTERN
                                  + maskname[maskname.rfind('/')+1:])


        ### VALIDATION ###
        for i in range(idx,len(images)):
                copyfile(images[i],
                          '/home/rubenh/projects/deeplab/models/research/deeplab/CAMELYON16_PREPROCESSING/Processed/%s/raw-data/validation/label-1/'%OUT_DIR_PATTERN
                          + images[i][images[i].rfind('/')+1:])

                maskname = images[i][:images[i].rfind('/') + 1] + 'mask_' + images[i][images[i].rfind('/') + 1:]
                match = [s for s in masks if maskname == s]

                if len(match):
                        copyfile(match[0],
                                  '/home/rubenh/projects/deeplab/models/research/deeplab/CAMELYON16_PREPROCESSING/Processed/%s/raw-data/validation/label-1/'%OUT_DIR_PATTERN
                                  + maskname[maskname.rfind('/')+1:])


        print("Finished copying")

