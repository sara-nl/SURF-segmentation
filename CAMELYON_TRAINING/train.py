from __future__ import absolute_import, division, print_function
import tensorflow as tf
import horovod.tensorflow as hvd
from pprint import pprint

from options import get_options
from utils import init, get_model_and_optimizer, setup_logger, log_training_step, log_validation_step
from data_utils import get_image_lists, get_train_and_val_dataset


def train_one_step(model, opt, x, y, step, loss_func, compression):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_func(y, logits)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape, device_sparse='/cpu:0', device_dense='/cpu:0', compression=compression) #  device_sparse='/cpu:0', device_dense='/cpu:0',
    grads = tape.gradient(loss, model.trainable_variables)

    opt.apply_gradients(zip(grads, model.trainable_variables))

    if step == 0:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

    pred = tf.argmax(logits, axis=-1)

    return loss, pred


def validate(opts, model, step, val_dataset, file_writer, metrics):
    """ Perform validation on the entire val_dataset """
    if hvd.local_rank() == 0 and hvd.rank() == 0:

        compute_loss, compute_accuracy, compute_miou, compute_auc = metrics
        val_loss, val_accuracy, val_miou, val_auc = [], [], [], []

        for image, label in val_dataset:
            val_pred_logits = model(image)
            val_pred = tf.math.argmax(val_pred_logits, axis=-1)
            val_loss.append(compute_loss(label, val_pred_logits))
            val_accuracy.append(compute_accuracy(label, val_pred))
            val_miou.append(compute_miou(label, val_pred))
            val_auc.append(compute_auc(label[:, :, :, 0], val_pred))

        val_loss = sum(val_loss) / len(val_loss)
        val_acc = sum(val_accuracy) / len(val_accuracy)
        val_miou = sum(val_miou) / len(val_miou)
        val_auc = sum(val_auc) / len(val_auc)

        image = tf.cast(255 * image, tf.uint8)
        mask = tf.cast(255 * label, tf.uint8)
        summary_predictions = tf.cast(val_pred * 255, tf.uint8)

        if len(summary_predictions.shape) == 3 and summary_predictions.shape[-1] != 1:
            summary_predictions = summary_predictions[:, :, :, None]

        log_validation_step(opts,file_writer, image, mask, step, summary_predictions, val_loss, val_acc, val_miou, val_auc)

        compute_accuracy.reset_states()
        compute_miou.reset_states()
        compute_auc.reset_states()



    return


def train(opts, model, optimizer, train_dataset, val_dataset, file_writer, compression):

    train_ds = train_dataset
    step = 0

    # Define metrics
    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    compute_accuracy = tf.keras.metrics.Accuracy()
    compute_miou = tf.keras.metrics.MeanIoU(num_classes=2)
    compute_auc = tf.keras.metrics.AUC()
    metrics = (compute_loss, compute_accuracy, compute_miou, compute_auc)

    while step < opts.num_steps:

        for x, y in train_ds:

            loss, pred = train_one_step(model, optimizer, x, y, step, compute_loss, compression)

            break
            if step % opts.log_every == 0 and step > 0:
                log_training_step(opts, model, file_writer, x, y, loss, pred, step, metrics)

            step += opts.batch_size * hvd.size()

            if step % opts.validate_every == 0:
                validate(opts, model, step, val_dataset, file_writer, metrics)

            if step > opts.num_steps:
                break

        if opts.hard_mining:
            # Bit ugly to define the function here, but it works
            def filter_hard_mining(image, mask):
                pred_logits = model(image)
                pred = tf.math.argmax(pred_logits, axis=-1)
                # Only select training samples with miou less then 0.95
                return tf.keras.metrics.MeanIoU(num_classes=2)(mask, pred) < 0.95

            train_ds = train_ds.filter(filter_hard_mining)


    validate(opts, model, step, val_dataset, file_writer, metrics)
    model.save('saved_model.h5')




if __name__ == '__main__':
    opts = get_options()
    pprint(vars(opts))
    # Run horovod init
    init(opts)
    file_writer = setup_logger(opts)

    train_dataset, val_dataset = get_train_and_val_dataset(opts)
    model, optimizer, compression = get_model_and_optimizer(opts)

    print('Preparing training...')
    train(opts, model, optimizer, train_dataset, val_dataset, file_writer, compression)
    print('Training is done')
