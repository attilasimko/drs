from tensorflow.python.keras.layers import Add, Conv3D, Input, Lambda, LeakyReLU, ReLU
from tensorflow.python.keras.models import Model
import tensorflow as tf
import tensorflow
from tensorflow.keras import layers
import numpy as np

from model.common import normalize, denormalize, pixel_shuffle, normalize_z, normalize_mean
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
    Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add , Concatenate, add, LeakyReLU
from tensorflow.keras.layers import Input, Conv3D, MaxPooling2D, ZeroPadding2D,\
    Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add , UpSampling2D, UpSampling3D, Conv2D, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2, l1
import numpy as np



def relu_range(x):
    x = tf.where(tf.keras.backend.greater(x, -1), x, -tf.keras.backend.ones_like(x))
    x = tf.where(tf.keras.backend.less(x, 1), x, tf.keras.backend.ones_like(x))
    return x

def residual_block(block_input, num_filters, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(block_input)
    x = BatchNormalization()(x)
    # x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([block_input, x])
    return x

def reduce_magnitude(tensor):
    import tf.keras.backend as K
    y_true_sqrt = tf.expand_dims(tf.math.sqrt(tf.reduce_max(tf.square(y_true_left), axis=4)), axis=4)
    return tf.math.divide_no_nan(tensor - t_mean[:, None, None, None], t_std[:, None, None, None])


def build_srresnet(inputs, num_filters=32, kspace=True, beta=1):

    if (kspace):
        k_in = Lambda(img_to_kspace)(inputs)
    else:
        k_in = inputs
    k_in = Lambda(normalize_k)(k_in)
    k_in = tf.multiply(k_in, beta)

    out = srresnet_backbone(k_in, num_filters)

    if (kspace):
        out = Conv2D(2, kernel_size=1, padding='same')(out)
        reg = tf.reduce_mean(tf.abs(out), axis=(1, 2, 3))
        out = Add()([k_in, out])
        out = Lambda(kspace_to_img)(out)
    else:
        out = Conv2D(1, kernel_size=1, padding='same')(out)
        k = Lambda(img_to_kspace)(out)
        k = Lambda(normalize_k)(k)
        reg = tf.reduce_mean(tf.abs(k), axis=(1, 2, 3))
        out = Add()([k_in, out])

    out = Lambda(normalize_z)(out)

    return Model(inputs, [out, reg])


def srresnet_backbone(inp, num_filters=32, batchnorm=True):
    x = Conv2D(8, kernel_size=3, padding='same')(inp)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_1 = LeakyReLU(0.2)(x)

    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_2 = LeakyReLU(0.2)(x)

    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_3 = LeakyReLU(0.2)(x)

    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_4 = LeakyReLU(0.2)(x)

    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_5 = LeakyReLU(0.2)(x)

    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_6 = LeakyReLU(0.2)(x)

    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Concatenate()([x, x_6])

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_5])
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = ReLU()(x) 

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_4])
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = ReLU()(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_3])
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = ReLU()(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_2])
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = ReLU()(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_1])
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    return x

def build_discriminator(x_input, num_filters=4):
    x = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(x_input)
    x = Lambda(reduce_magnitude)(x)

    return Model(x_input, x)




def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv3D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='Conv3D_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='Conv3D_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='Conv3D_1_scale_2')
        x = upsample_1(x, 2, name='Conv3D_2_scale_2')

    return x
