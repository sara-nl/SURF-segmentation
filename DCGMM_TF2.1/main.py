import tensorflow as tf
import numpy as np
from pprint import pprint
import pdb
from model import DCGMM
from options import get_options
from dataset_utils import get_train_and_val_dataset, get_template_and_image_dataset
from utils import get_model_and_optimizer, setup_normalizing_run
from logging_utils import setup_logger
from train import train
from eval import eval_mode


if __name__ == '__main__':
    opts = get_options()
    pprint(vars(opts))

    # Start running training
    if not opts.eval_mode:
        tb_logger, logdir = setup_logger(opts)
        e_step, m_step, optimizer = get_model_and_optimizer(opts)
        train_dataset, val_dataset = get_train_and_val_dataset(opts)
        train(opts, e_step, m_step, optimizer, train_dataset, val_dataset, tb_logger,logdir)

    # Start running inference
    else:
        setup_normalizing_run(opts)
        e_step, m_step, _ = get_model_and_optimizer(opts)
        template_dataset, image_dataset = get_template_and_image_dataset(opts)
        eval_mode(opts, e_step, m_step, template_dataset, image_dataset)
