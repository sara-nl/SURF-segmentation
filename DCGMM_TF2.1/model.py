import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
import tensorflow_probability as tfp
import pdb

def CNN(opts, input_tensor=None):
    """ Initializes the CNN backbone of the DCGMM model.
        Arguments:
        - input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
        - input_shape: shape of input image. format HxWxC
        - num_clusters: number of clusters to use
    """
    if input_tensor is None:
        input_shape = (opts.img_size, opts.img_size, 1)
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    x1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                data_format='channels_last', strides=(1, 1), activation='relu')(img_input)

    x2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                data_format='channels_last', strides=(1, 1), activation='relu')(x1)

    x2 = MaxPool2D(pool_size=(2, 2), padding='same')(x2)

    x3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                data_format='channels_last', strides=(1, 1), activation='relu')(x2)

    x4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                data_format='channels_last', strides=(1, 1), activation='relu')(x3)

    x4 = MaxPool2D(pool_size=(2, 2), padding='same')(x4)

    x5 = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                data_format='channels_last', strides=(1, 1), activation='relu')(x4)

    # Upscale the H and W dimension by a factor 2
    output_shape = x5.shape.as_list()
    output_shape[1] *= 2
    output_shape[2] *= 2
    x5 = tf.image.resize(x5, output_shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    x6 = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                data_format='channels_last', strides=(1, 1), activation='relu')(x5)

    # Upscale the H and W dimension by a factor 2
    output_shape = x6.shape.as_list()
    output_shape[1] *= 2
    output_shape[2] *= 2
    x6 = tf.image.resize(x6, output_shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    x7 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                data_format='channels_last', strides=(1, 1), activation='relu')(x6)

    x8 = Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                data_format='channels_last', strides=(1, 1), activation='relu')(x7)

    x9 = Conv2D(filters=opts.num_clusters, kernel_size=(3, 3), padding='same',
                data_format='channels_last', strides=(1, 1), activation='relu')(x8)

    gamma = Softmax(axis=-1)(x9)

    model = Model(img_input, gamma, name='CNN_backbone')
    return model


def GMM_M_Step(X_hsd, gamma, opts, name='GMM_Statistics', **kwargs):
    

    D, s, h = tf.split(X_hsd, [1, 1, 1], axis=3)

    WXd = tf.multiply(gamma, tf.tile(D, [1, 1, 1, opts.num_clusters]))
    WXa = tf.multiply(gamma, tf.tile(h, [1, 1, 1, opts.num_clusters]))
    WXb = tf.multiply(gamma, tf.tile(s, [1, 1, 1, opts.num_clusters]))
    S = tf.reduce_sum(tf.reduce_sum(gamma, axis=1), axis=1)
    S = tf.add(S, tf.keras.backend.epsilon())
    S = tf.reshape(S, [opts.batch_size, opts.num_clusters])

    M_d = tf.divide(tf.reduce_sum(tf.reduce_sum(WXd, axis=1), axis=1), S)
    M_a = tf.divide(tf.reduce_sum(tf.reduce_sum(WXa, axis=1), axis=1), S)
    M_b = tf.divide(tf.reduce_sum(tf.reduce_sum(WXb, axis=1), axis=1), S)

    mu = tf.split(tf.concat([M_d, M_a, M_b], axis=0), opts.num_clusters, 1)

    Norm_d = tf.math.squared_difference(D, tf.reshape(M_d, [opts.batch_size,1,1, opts.num_clusters]))
    Norm_h = tf.math.squared_difference(h, tf.reshape(M_a, [opts.batch_size,1,1, opts.num_clusters]))
    Norm_s = tf.math.squared_difference(s, tf.reshape(M_b, [opts.batch_size,1,1, opts.num_clusters]))

    WSd = tf.multiply(gamma, Norm_d)
    WSh = tf.multiply(gamma, Norm_h)
    WSs = tf.multiply(gamma, Norm_s)

    S_d = tf.sqrt(tf.divide(tf.reduce_sum(tf.reduce_sum(WSd, axis=1), axis=1), S))
    S_h = tf.sqrt(tf.divide(tf.reduce_sum(tf.reduce_sum(WSh, axis=1), axis=1), S))
    S_s = tf.sqrt(tf.divide(tf.reduce_sum(tf.reduce_sum(WSs, axis=1), axis=1), S))

    std = tf.split(tf.concat([S_d, S_h, S_s], axis=0), opts.num_clusters, 1)

    dist = [tfp.distributions.MultivariateNormalDiag(tf.reshape(mu[k], [opts.batch_size,1,1, 3]),
                                                     tf.reshape(std[k], [opts.batch_size,1,1, 3])) for k in range(opts.num_clusters)]

    pi = tf.split(gamma, opts.num_clusters, axis=-1)

    prob0 = [tf.multiply(tf.squeeze(dist[k].prob(X_hsd)), tf.squeeze(pi[k])) for k in range(opts.num_clusters)]

    prob = tf.convert_to_tensor(prob0, dtype=tf.float32)
    prob = tf.minimum(tf.add(tf.reduce_sum(prob, axis=0), tf.keras.backend.epsilon()),
                      tf.constant(1.0, tf.float32))
    log_prob = tf.negative(tf.math.log(prob))
    ll = tf.reduce_mean(log_prob)
    return ll, mu, std


class DCGMM(tf.keras.Model):
    def __init__(self, opts):
        super(DCGMM, self).__init__()
        self.opts = opts
        self.e_step = CNN(self.opts)
        self.m_step = GMM_M_Step

    def __call__(self, rgb_img, hsd_img):
        """ Call the model, perform an E-step and a M-step """

        if isinstance(hsd_img, np.ndarray):
            hsd_img = tf.convert_to_tensor(hsd_img, dtype=tf.float32)

        h, s, d = tf.split(hsd_img, 3, axis=-1)

        gamma = self.e_step(d)
        ll, mu, std = self.m_step(hsd_img, gamma)

        return ll, mu, std

    def e_step(self, rgb_img, hsd_img):
        """ Perform only the E-step """
        if isinstance(hsd_img, np.ndarray):
            hsd_img = tf.convert_to_tensor(hsd_img, dtype=tf.float32)

        h, s, d = tf.split(hsd_img, 3, axis=-1)

        gamma = self.e_step(d)
        return gamma

    def eval(self, rgb_img, hsd_img):
        """ Perform a step needed for inference """

        if isinstance(hsd_img, np.ndarray):
            hsd_img = tf.convert_to_tensor(hsd_img, dtype=tf.float32)

        h, s, d = tf.split(hsd_img, 3, axis=-1)

        gamma = self.e_step(d)
        _, mu, std = self.m_step(hsd_img, gamma)

        return mu, std, gamma


if __name__ == '__main__':
    import hasel
    import imageio

    dcgmm = DCGMM()

    rgb_img = imageio.imread('images/0', pilmode='RGB')
    hsd_img = hasel.rgb2hsl(rgb_img)[None, :, :, :]

    h, s, d = dcgmm(rgb_img, hsd_img)
