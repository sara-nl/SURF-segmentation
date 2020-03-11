import tensorflow as tf
import pdb

from logging_utils import log_training_step, save_model


def train_one_step(opts, e_step, m_step, optimizer, img_rgb, img_hsd, step):
    """ Perform one training step. """

    if opts.normalize_imgs:
        img_rgb = (img_rgb * 2) - 1.
        img_hsd = (img_hsd * 2) - 1.

    # First split into the three channels. Necessary for the E-step, which only takes the 'D' channel
    _, _, d_channel = tf.split(img_hsd, 3, axis=-1)

    pdb.set_trace()

    with tf.GradientTape() as tape:
        gamma = e_step(d_channel)
        ll, mu, std = m_step(img_hsd, gamma, opts.num_clusters)

    grads = tape.gradient(ll, e_step.trainable_variables)
    optimizer.apply_gradients(zip(grads, e_step.trainable_variables))

    return ll, gamma, mu, std


def train(opts, e_step, m_step, optimizer, train_dataset, val_dataset, file_writer):
    train_ds = train_dataset
    step = 0

    while step < opts.num_steps:

        img_rgb, img_hsd = train_ds.get_next_batch()

        ll, gamma, mu, std = train_one_step(opts, e_step, m_step, optimizer, img_rgb, img_hsd, step)

        step += opts.batch_size

        if step > opts.num_steps:
            break

        if step % opts.log_every == 0:
            log_training_step(opts, step, ll, img_rgb, img_hsd, gamma, file_writer)

        if step % opts.save_every == 0:
            save_model(opts, step, e_step)

        if step % opts.validate_every == 0:
            # TODO: make validate function
            # validate(opts, step, e_step, m_step, val_dataset, file_writer)
            pass

    e_step.save(os.path.join(opts.logdir, f'checkpoint_{step}'))


def validate(opts, step, e_step, m_step, val_dataset, file_writer):
    raise NotImplementedError("This has not yet been implemented")

