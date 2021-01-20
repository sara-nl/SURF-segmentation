import numpy as np
import tensorflow as tf
# import hasel
import matplotlib.pyplot as plt
import shutil
import os
import pdb
from model import CNN, GMM_M_Step

def RGB2HSD_legacy(X):
    # eps = np.finfo(float).eps
    eps=1e-7
    X[np.where(X == 0.0)] = eps

    OD = -np.log(X / 1.0)
    D = np.mean(OD, X.ndim - 1)
    D[np.where(D == 0.0)] = eps

    cx = OD[:, :, :, 0] / (D) - 1.0
    cy = (OD[:, :, :, 1] - OD[:, :, :, 2]) / (np.sqrt(3.0) * D)

    D = np.expand_dims(D, 3)
    cx = np.expand_dims(cx, 3)
    cy = np.expand_dims(cy, 3)

    X_HSD = np.concatenate((D,cx, cy), 3)
    return X_HSD


def HSD2RGB_legacy(X_HSD):
    X_HSD_0, X_HSD_1, X_HSD_2 = tf.split(X_HSD, [1, 1, 1], axis=3)
    D_R = (X_HSD_1 + 1) * X_HSD_0
    D_G = 0.5 * X_HSD_0 * (2 - X_HSD_1 + tf.sqrt(tf.constant(3.0)) * X_HSD_2)
    D_B = 0.5 * X_HSD_0 * (2 - X_HSD_1 - tf.sqrt(tf.constant(3.0)) * X_HSD_2)

    X_OD = tf.concat([D_R, D_G, D_B], 3)
    X_RGB = 1.0 * tf.exp(-X_OD)
    return X_RGB


def HSD2RGB_Numpy_legacy(X_HSD):
    X_HSD_0 = X_HSD[..., 0]
    X_HSD_1 = X_HSD[..., 1]
    X_HSD_2 = X_HSD[..., 2]
    D_R = np.expand_dims(np.multiply(X_HSD_1 + 1, X_HSD_0), -1)
    D_G = np.expand_dims(np.multiply(0.5 * X_HSD_0, 2 - X_HSD_1 + np.sqrt(3.0) * X_HSD_2), -1)
    D_B = np.expand_dims(np.multiply(0.5 * X_HSD_0, 2 - X_HSD_1 - np.sqrt(3.0) * X_HSD_2), -1)

    X_OD = np.concatenate((D_R, D_G, D_B), axis=-1)
    X_RGB = 1.0 * np.exp(-X_OD)
    return X_RGB


def image_dist_transform(opts, img_hsd, mu, std, gamma, mu_tmpl, std_tmpl):
    """ Given a mu and std of an image and template, apply the color normalization """

    # alle mu std, (4,1,3)

    img_norm = np.empty((opts.batch_size,opts.img_size, opts.img_size, 3, opts.num_clusters))
    
    mu  = np.reshape(mu, [mu.shape[0] ,opts.batch_size,1,1,3])
    std = np.reshape(std,[std.shape[0],opts.batch_size,1,1,3])
    mu_tmpl  = np.reshape(mu_tmpl, [mu_tmpl.shape[0] ,opts.batch_size,1,1,3])
    std_tmpl = np.reshape(std_tmpl,[std_tmpl.shape[0],opts.batch_size,1,1,3])
    for c in range(0, opts.num_clusters):
        img_normalized = np.divide(np.subtract(np.squeeze(img_hsd), mu[c, ...]), std[c, ...])
        img_univar = np.add(np.multiply(img_normalized, std_tmpl[c, ...]), mu_tmpl[c, ...])
        # img_univar = np.add(np.zeros_like(img_norm), mu[c,...])
        img_norm[..., c] = np.multiply(img_univar, np.tile(np.expand_dims(np.squeeze(gamma[..., c]), axis=-1), (1, 1, 3)))

    
    img_norm = np.sum(img_norm, axis=-1)
    # Apply the triangular restriction to cxcy plane in HSD color coordinates
    img_norm = np.split(img_norm, 3, axis=-1)
    
    img_norm[1] = np.maximum(np.minimum(img_norm[1], 2.0), -1.0)
    img_norm = np.squeeze(np.swapaxes(np.asarray(img_norm), 0, -1))

    ## Transfer from HSD to RGB color coordinates
    if opts.legacy_conversion:
        img_norm = HSD2RGB_Numpy_legacy(img_norm)
        img_norm = np.minimum(img_norm, 1.0)
        img_norm = np.maximum(img_norm, 0.0)
        img_norm *= 255
        img_norm = img_norm.astype(np.uint8)
    else:
        # img_norm = hasel.hsl2rgb(img_norm)
        print("Not doing Hsl transform")

    return img_norm


def get_model_and_optimizer(opts):
    """ Load the model and optimizer """
    m_step = GMM_M_Step

    if not opts.eval_mode:
        e_step = CNN(opts)
        e_step.build(input_shape=(opts.img_size, opts.img_size, 1))
        opt = tf.optimizers.Adam(0.0001, epsilon=1e-7)
    else:
        e_step = tf.saved_model.load(opts.load_path)
        opt = None

    return e_step, m_step, opt


def setup_normalizing_run(opts):
    """ Setup a run to color-normalize a dataset of images """
    shutil.rmtree(opts.save_path, ignore_errors=True)
    os.makedirs(opts.save_path, exist_ok=True)
