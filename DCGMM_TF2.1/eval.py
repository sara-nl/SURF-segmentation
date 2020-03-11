import tensorflow as tf
import numpy as np
import matplotlib
import os

from utils import image_dist_transform


def deploy(opts, e_step, m_step, img_rgb, img_hsd):
    """ Perform a step needed for inference """

    if opts.normalize_imgs:
        img_rgb = (img_rgb * 2) - 1.
        img_hsd = (img_hsd * 2) - 1.

    # First split into the three channels. Necessary for the E-step, which only takes the 'D' channel
    _, _, d_channel = tf.split(img_hsd, 3, axis=-1)

    gamma = e_step(d_channel)
    _, mu, std = m_step(img_hsd, gamma, opts.num_clusters)

    mu = np.asarray(mu)
    mu = np.swapaxes(mu, 1, 2)  # -> dim: [ClustrNo x 1 x 3]
    std = np.asarray(std)
    std = np.swapaxes(std, 1, 2)  # -> dim: [ClustrNo x 1 x 3]

    return mu, std, gamma


def eval_mode(opts, e_step, m_step, template_dataset, image_dataset):
    """ Normalize entire images """

    # Determine mu and std of the template first
    mu_tmpl = 0
    std_tmpl = 0
    N = 0

    while template_dataset.batch_offset < len(template_dataset):

        img_rgb, img_hsd = template_dataset.get_next_batch()
        mu, std, gamma = deploy(opts, e_step, m_step, img_rgb, img_hsd)

        N += 1
        mu_tmpl = (N - 1) / N * mu_tmpl + 1 / N * mu
        std_tmpl = (N - 1) / N * std_tmpl + 1 / N * std

    i = 0
    while image_dataset.batch_offset < len(image_dataset):

        img_rgb, img_hsd = image_dataset.get_next_image()
        mu, std, gamma = deploy(opts, e_step, m_step, img_rgb, img_hsd)

        img_norm = image_dist_transform(opts, img_hsd, mu, std, gamma, mu_tmpl, std_tmpl)
        matplotlib.image.imsave(os.path.join(opts.save_path, f'{i}.png'), img_norm)

        i += 1





