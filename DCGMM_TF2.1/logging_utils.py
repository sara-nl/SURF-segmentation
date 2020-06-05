import os
import tensorflow as tf
import datetime
from time import gmtime, strftime
import os
import shutil

def setup_logger(opts):
    """ Setup the tensorboard writer """
    # Sets up a timestamped log directory.
    if opts.dataset == "17":
        logdir = "logs/train_data/" + str(opts.img_size) + '-tr' + ''.join(map(str, opts.train_centers)) + \
                 '-val' + ''.join(map(str, opts.val_centers))
    else:
        logdir = os.path.join('logs', f'CAMELYON16_{opts.img_size}', strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    if opts.debug:
        logdir += '-_debug'

    if opts.log_dir:
        logdir = opts.log_dir

    shutil.rmtree(logdir, ignore_errors=True)
    os.makedirs(logdir, exist_ok=True)
    file_writer = tf.summary.create_file_writer(logdir)

    return file_writer, logdir


def log_training_step(opts, step, ll, img_rgb, img_hsd, gamma, file_writer):
    """ Log a single training step """

    _, _, d_channel = tf.split(img_hsd, 3, axis=-1)

    with file_writer.as_default():
        # Log the loss
        tf.summary.scalar('Training loss', ll, step)

        img_rgb = tf.cast(img_rgb * 255, tf.uint8)
        img_hsd = tf.cast(img_hsd * 255, tf.uint8)
        d_channel = tf.cast(d_channel * 255, tf.uint8)

        tf.summary.image("1. Training input_image", img_rgb, step, max_outputs=10)
        tf.summary.image("2. Training hsd_image", img_hsd, step, max_outputs=10)

        ClsLbl = tf.cast(tf.math.argmax(gamma, axis=-1), tf.float32)
        ColorTable = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
        colors = tf.constant(ColorTable, dtype=tf.float32)

        mask = tf.tile(tf.expand_dims(ClsLbl, axis=3), [1, 1, 1, 3])
        for k in range(0, opts.num_clusters):
            ClrTmpl = tf.einsum('anmd,df->anmf', tf.expand_dims(tf.ones_like(ClsLbl), axis=3),
                                tf.reshape(colors[k, ...], [1, 3]))
            mask = tf.where(tf.equal(mask, k), ClrTmpl, mask)

        tf.summary.image("3. Training gamma_image", mask, step, max_outputs=10)
        tf.summary.image("4. Training density_image", d_channel, step, max_outputs=10)

    file_writer.flush()


def save_model(opts, step, e_step, logdir):
    saving_path = os.path.join(logdir, f'checkpoint_{step}')
    e_step.save(saving_path)
