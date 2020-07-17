import tensorflow as tf
import argparse


def get_options():
    """ Argument parsing options"""

    parser = argparse.ArgumentParser(description='Multi - GPU TensorFlow DeeplabV3+ model',
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
    parser.add_argument('--fp16_allreduce', action='store_true', help='Reduce to FP16 precision for gradient all reduce')

    # == Dataset and path options ==
    parser.add_argument('--train_centers', nargs='+', default=[-1], type=int,
                        help='Centers for training. Use -1 for all, otherwise 2 3 4 eg.')
    parser.add_argument('--val_centers', nargs='+', default=[-1], type=int,
                        help='Centers for validation. Use -1 for all, otherwise 2 3 4 eg.')
    parser.add_argument('--hard_mining', action='store_true', help='Use hard mining or not')
    parser.add_argument('--slide_path', type=str, help='Folder of where the training data whole slide images are located', default=None)
    parser.add_argument('--mask_path', type=str, help='Folder of where the training data whole slide images masks are located', default=None)
    parser.add_argument('--valid_slide_path', type=str, help='Folder of where the validation data whole slide images are located', default=None)
    parser.add_argument('--valid_mask_path', type=str, help='Folder of where the validation data whole slide images masks are located', default=None)
    parser.add_argument('--weights_path', type=str, help='Folder where the pre - trained weights is located', default=None)
    parser.add_argument('--bb_downsample', type=int, help='Level to use for the bounding box construction as downsampling level of whole slide image', default=7)
    parser.add_argument('--slide_format', type=str, help='In which format the whole slide images are saved.', default='tif')
    parser.add_argument('--mask_format', type=str, help='In which format the masks are saved.', default='tif')
    parser.add_argument('--log_image_path', type=str, help='Path of savepath of downsampled image with processed rectangles on it.', default='.')
    parser.add_argument('--batch_tumor_ratio', type=float, help='The ratio of the batch that contains tumor', default=1)
    
    

    # == Data augmentation options ==
    parser.add_argument('--log_dir', type=str, help='Folder of where the logs are saved', default=None)
    parser.add_argument('--log_every', type=int, default=128, help='Log every X steps during training')
    parser.add_argument('--validate_every', type=int, default=2048, help='Run the validation dataset every X steps')
    parser.add_argument('--debug', action='store_true', help='If running in debug mode, only uses 100 images')

    # == Redundant ==
    parser.add_argument('--pos_pixel_weight', type=int, default=1)
    parser.add_argument('--neg_pixel_weight', type=int, default=1)

    opts = parser.parse_args()

    opts.cuda = not opts.no_cuda
    opts.tf_version = f'{tf.__version__}'

    return opts
