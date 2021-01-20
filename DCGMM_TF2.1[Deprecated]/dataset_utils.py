import tensorflow as tf
import pdb
from glob import glob
import os
from sklearn.utils import shuffle
# import hasel
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from utils import RGB2HSD_legacy
import pprint

def get_image_lists(opts):
    """ Get the image lists"""

    if opts.dataset == "17":
        image_list, val_image_list = load_camelyon_17(opts)
    else:
        image_list, val_image_list = load_paths(opts)
    
    print('Found', len(image_list), 'training images')
    print('Found', len(val_image_list), 'validation images')
    return image_list, val_image_list


def load_data(image_path, opts):
    """ Load the image. Augmentation options can be added here """
    img_rgb, img_hsd = get_image(image_path, opts)

    return img_rgb, img_hsd


def get_image(image_path, opts):
    img_rgb = imageio.imread(image_path, pilmode='RGB')

    img_hsd_legacy = RGB2HSD_legacy(img_rgb[np.newaxis, :] / 255.)[0]
    # Normalize this image between 0 and 1
    # img_hsd_legacy = (img_hsd_legacy - np.min(img_hsd_legacy)) / np.ptp(img_hsd_legacy)

    # img_hsd = hasel.rgb2hsl(img_rgb)

    # if opts.debug:

    #     combined = np.hstack([img_hsd, img_hsd_legacy])

    #     plt.imshow(combined)
    #     plt.title('Left: hasel conversion. Right: legacy conversion')
    #     filename = image_path.split('/')[-1].split('_')[-1] + '.png'
    #     plt.savefig(os.path.join('images', filename))
    #     plt.close()

    #     combined_D = combined[:, :, -1]
    #     plt.imshow(combined_D, cmap='gray')
    #     plt.title('Left: hasel conversion. Right: legacy conversion')
    #     filename = image_path.split('/')[-1].split('_')[-1] + '_D.png'
    #     plt.savefig(os.path.join('images', filename))
    #     plt.close()

    if opts.legacy_conversion:
        img_hsd = img_hsd_legacy

    img_rgb = img_rgb / 255.

    return img_rgb.astype(np.float32), img_hsd.astype(np.float32)


def get_train_and_val_dataset(opts):
    """ Get the training and validation dataset"""

    train_dataset = CustomDCGMMDataset(opts, is_train=True, is_valid=False, is_template=False, is_eval=False)
    val_dataset = CustomDCGMMDataset(opts, is_train=False, is_valid=True, is_template=False, is_eval=False)

    return train_dataset, val_dataset


def get_template_and_image_dataset(opts):
    """ Get a dataset representing the template image(s) """
    template_dataset = CustomDCGMMDataset(opts, is_train=False, is_valid=False, is_template=True, is_eval=False)
    image_dataset = CustomDCGMMDataset(opts, is_train=False, is_valid=False, is_template=False, is_eval=True)

    return template_dataset, image_dataset


def load_paths(opts):
    """  Load the camelyon16 dataset """

    image_list = [x for x in sorted(glob(os.path.join(opts.train_path,'*'), recursive=True)) if 'mask' not in x]
    print(f"path {os.path.join(opts.train_path,'*')}")
    image_list = shuffle(image_list)

    if opts.debug:
        image_list = image_list[0:100]
    
    if opts.val_split:
        val_split = int(len(image_list) * (1 - opts.val_split))
        val_image_list = image_list[val_split:]
        image_list = image_list[:val_split]
    else:
        val_image_list = [x for x in sorted(glob(os.path.join(opts.valid_path,'*'), recursive=True)) if 'mask' not in x]

    return image_list, val_image_list

def load_camelyon_17(opts):
    """ Load the camelyon17 dataset """
    image_list = [x for c in opts.train_centers for x in sorted(glob(os.path.join(str(opts.train_path).replace('center_XX', f'center_{c}'),'*'), recursive=True)) if 'mask' not in x]
    
    mask_list = [x for c in opts.train_centers for x in sorted(glob(os.path.join(str(opts.train_path).replace('center_XX', f'center_{c}'),'*'), recursive=True)) if'mask' in x]
    
    sample_weight_list = [1.0] * len(image_list)

    # If validating on everything, 00 custom
    if opts.val_centers == [1, 2, 3, 4]:
        val_split = int(len(image_list) * (1-opts.val_split))
        val_image_list = image_list[val_split:]
        val_mask_list = mask_list[val_split:]
        sample_weight_list = sample_weight_list[:val_split]
        image_list = image_list[:val_split]
        mask_list = mask_list[:val_split]

        idx = [np.asarray(Image.open(x))[:, :, 0] / 255 for x in val_mask_list]
        num_pixels = opts.img_size ** 2
        valid_idx = [((num_pixels - np.count_nonzero(x)) / num_pixels) >= 0.2 for x in idx]
        valid_idx = [i for i, x in enumerate(valid_idx) if x]

        val_image_list = [val_image_list[i] for i in valid_idx]
        val_mask_list = [val_mask_list[i] for i in valid_idx]

        val_image_list, val_mask_list = shuffle(val_image_list, val_mask_list)

    else:
        val_image_list = [x for c in opts.val_centers for x in
                          sorted(glob(os.path.join(opts.valid_path.replace('center_XX', f'center_{c}'),'*'), recursive=True)) if
                          'mask' not in x]
        val_mask_list = [x for c in opts.val_centers for x in
                         sorted(glob(os.path.join(opts.valid_path.replace('center_XX', f'center_{c}'),'*'), recursive=True)) if
                         'mask' in x]
        
    # return image_list, mask_list, val_image_list, val_mask_list, sample_weight_list
    return image_list, val_image_list


class CustomDCGMMDataset:

    def __init__(self, opts, is_train=True, is_valid=False, is_template=False, is_eval=False):
        self.opts = opts

        assert int(is_train) + int(is_valid) + int(is_template) + int(is_eval) == 1, "Can only be one type of dataset"

        if is_train:
            image_list, _ = get_image_lists(opts)
            self.image_list = image_list
        elif is_valid:
            _, val_image_list = get_image_lists(opts)
            self.image_list = val_image_list
        elif is_template:
            self.image_list = [x for x in sorted(glob(os.path.join(opts.template_path,'*'), recursive=True)) if 'mask' not in x]
        elif is_eval:
            self.image_list = [x for x in sorted(glob(os.path.join(opts.images_path,'*'), recursive=True)) if 'mask' not in x]

        if self.opts.debug and self.image_list:
            self.image_list = self.image_list[:10]
        
        self.batch_offset = 0
        self.epochs_completed = 0
        self.current_epoch = 0

    def __len__(self):
        return len(self.image_list)

    def get_next_batch(self):

        start = self.batch_offset
        # Epoch completed, reshuffle data
        if self.batch_offset >= len(self.image_list) - self.opts.batch_size:
            print("Epoch completed")
            self.image_list = shuffle(self.image_list)
            start             = 0
            self.batch_offset = 0

        images = [load_data(path, self.opts) for path in self.image_list[start:start + self.opts.batch_size]]
        rgb_imgs, hsd_imgs = zip(*images)

        self.batch_offset += self.opts.batch_size

        return np.array(rgb_imgs), np.array(hsd_imgs), self.image_list[start:start + self.opts.batch_size]

    def get_next_image(self):
        """ Similar to get_next_batch, only prevents looping over a dataset """
        img_rgb, img_hsd = load_data(self.image_list.pop(0), self.opts)
        return np.array(img_rgb)[None, :], np.array(img_hsd)[None, :]


if __name__ == '__main__':
    from options import get_options

    opts = get_options()
    opts.train_path = 'images'

    dataloader = CustomDCGMMDataset(opts)
    dataloader.get_next_batch()
