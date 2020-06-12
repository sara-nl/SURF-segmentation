import argparse
import tensorflow as tf
import os
import datetime
from time import gmtime, strftime


def get_options():
    """ Argument parsing options"""

    parser = argparse.ArgumentParser(description='TF2 DCGMM model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--eval_mode', action='store_true',
                        help='Run in evaluation mode. If false, training mode is activated')

    parser.add_argument('--img_size', type=int, default=256, help='Image size to use')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size to use')
    parser.add_argument('--epochs', type=int, default=1,    help='Number of epochs for training.')
    parser.add_argument('--num_clusters', type=int, default=4, help='Number of tissue classes to use in DCGMM modelling')

    # Dataset and path options
    parser.add_argument('--dataset', type=str, default="16",
                        help='Which dataset to use. "16" for CAMELYON16 or "17" for CAMELYON17')
    parser.add_argument('--train_centers', nargs='+', default=[-1], type=int,
                        help='Centers for training. Use -1 for all')
    parser.add_argument('--val_centers', nargs='+', default=[-1], type=int,
                        help='Centers for validation. Use -1 for all')
    parser.add_argument('--train_path'  , type=str, help='Folder of where the training data is located', default=None)
    parser.add_argument('--valid_path', type=str, help='Folder where the validation data is located', default=None)
    parser.add_argument('--logdir', type=str, help='Folder where to log tensorboard and model checkpoints',
                        default='logs')
    parser.add_argument('--template_path', type=str, help='Folder where template images are stored for deployment.', default='template')
    parser.add_argument('--images_path', type=str, help='Path where images to normalize are located', default='images')
    parser.add_argument('--load_path', type=str, help='Path where to load model from',
                        default='logs/train_data')
    parser.add_argument('--save_path', type=str, default='0', help='Where to save normalized images')

    # Data augmentation options
    parser.add_argument('--legacy_conversion', action='store_true', help='Legacy HSD conversion', default=True)
    parser.add_argument('--normalize_imgs', action='store_true', help='Normalize images between -1 and 1', default=False)

    parser.add_argument('--log_every', type=int, default=100, help='Log every X steps during training')
    parser.add_argument('--save_every', type=int, default=1000, help='Save a checkpoint every X steps')
    parser.add_argument('--debug', action='store_true', help='If running in debug mode (only 10 images)')
    parser.add_argument('--val_split', type=float, default=0.15)

    opts = parser.parse_args()

    # Default the paths to work for the owner of this repo
    if opts.dataset == '16' and opts.train_path is None:
        opts.train_path = f'/nfs/managed_datasets/CAMELYON16/pro_patch_positive_{opts.img_size}'
        opts.valid_path = f'/nfs/managed_datasets/CAMELYON16/pro_patch_positive_{opts.img_size}'
        # opts.train_path = 'images'
        # opts.valid_path = 'images'
    elif opts.dataset == '17' and opts.train_path is None:
        opts.train_path = f'/nfs/managed_datasets/CAMELYON17/training/center_XX/patches_positive_{opts.img_size}'
        opts.valid_path = f'/nfs/managed_datasets/CAMELYON17/training/center_XX/patches_positive_{opts.img_size}'

    if opts.dataset == "17" and opts.logdir is None:
        opts.logdir = os.path.join('logs', f'CAMELYON17_{opts.img_size}',
                                   strftime("%Y-%m-%d %H:%M:%S", gmtime())) + '-tr' + ''.join(
            map(str, opts.train_centers)) + '-val' + ''.join(map(str, opts.val_centers))

    elif opts.dataset == '16' and opts.logdir is None:
        opts.logdir = os.path.join('logs', f'CAMELYON16_{opts.img_size}', strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    if opts.eval_mode:
        assert opts.template_path is not None, 'A path to the template image(s) should be provided'
        assert opts.load_path is not None, 'A path where a saved model is located should be provided'
        assert opts.save_path is not None, 'A path where normalized images should be saved should be provided'

    opts.tf_version = f'{tf.__version__}'

    return opts
