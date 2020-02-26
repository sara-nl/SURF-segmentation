import tensorflow as tf
import argparse


def get_options():
    """ Argument parsing options"""

    parser = argparse.ArgumentParser(description='TensorFlow DeeplabV3+ model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # == Memory time consumption ==
    parser.add_argument('--img_size', type=int, default=1024, help='Image size to use')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size to use')
    parser.add_argument('--num_steps', type=int, default=50000,
                        help='Number of steps for training. A single step is defined as one image. So a batch of 2 consists of 2 steps')
    parser.add_argument('--val_split', type=float, default=0.15, help='Part of images that is used as validation dataset, validating on all images')


    # == GPU and multi worker options ==
    parser.add_argument('--no_cuda', action='store_true', help='Use CUDA or not')
    parser.add_argument('--horovod', action='store_true', help='Distributed training via horovod', default=True)
    parser.add_argument('--fp16_allreduce', action='store_true', help='Reduce to FP16 precision')

    # == Dataset and path options ==
    parser.add_argument('--dataset', type=str, default="17",
                        help='Which dataset to use. "16" for CAMELYON16 or "17" for CAMELYON17')
    parser.add_argument('--train_centers', nargs='+', default=[-1], type=int,
                        help='Centers for training. Use -1 for all, otherwise 2 3 4 eg.')
    parser.add_argument('--val_centers', nargs='+', default=[-1], type=int,
                        help='Centers for validation. Use -1 for all, otherwise 2 3 4 eg.')
    parser.add_argument('--hard_mining', action='store_true', help='Use hard mining or not')
    parser.add_argument('--train_path', type=str, help='Folder of where the training data is located', default=None)
    parser.add_argument('--valid_path', type=str, help='Folder where the validation data is located', default=None)

    # == Data augmentation options ==
    parser.add_argument('--flip', action='store_true', help='Flip images for data augmentation')
    parser.add_argument('--random_crop', action='store_true', help='Randomly crop images for data augmentation')
    parser.add_argument('--normalize_imgs', action='store_true', help='Normalize images')

    parser.add_argument('--log_dir', type=str, help='Folder of where the logs are saved', default=None)
    parser.add_argument('--log_every', type=int, default=128, help='Log every X steps during training')
    parser.add_argument('--validate_every', type=int, default=2048, help='Run the validation dataset every X steps')
    parser.add_argument('--debug', action='store_true', help='If running in debug mode')

    # == Redundant ==
    parser.add_argument('--pos_pixel_weight', type=int, default=1)
    parser.add_argument('--neg_pixel_weight', type=int, default=1)

    opts = parser.parse_args()

    # == Default the paths to work for the owner of this repo ==
    if opts.dataset == '17' and opts.train_path is None:
        opts.train_path = f'/nfs/managed_datasets/CAMELYON17/training/center_XX/patches_positive_{opts.img_size}'
        opts.valid_path = f'/nfs/managed_datasets/CAMELYON17/training/center_XX/patches_positive_{opts.img_size}'

    if opts.dataset == '16' and opts.train_path is None:
        opts.train_path = f'/nfs/managed_datasets/CAMELYON16/pro_patch_positive_{opts.img_size}'
        opts.valid_path = f'/nfs/managed_datasets/CAMELYON16/pro_patch_positive_{opts.img_size}'

    opts.cuda = not opts.no_cuda
    opts.tf_version = f'{tf.__version__}'

    return opts
