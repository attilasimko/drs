from encodings import normalize_encoding
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout,\
    Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add , Concatenate, add, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import softmax
import math
import numpy as np

def relu_range(x):
    import tensorflow
    from tensorflow.keras import backend as K
    x = tensorflow.where(K.greater_equal(x, -2), x, -2 * K.ones_like(x))
    x = tensorflow.where(K.less_equal(x, 2), x, 2 * K.ones_like(x))
    return x

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, Lambda, UpSampling2D, Conv2DTranspose
from tensorflow.python.keras.layers import PReLU, ReLU



def residual_block(block_input, num_filters, momentum=0.8, batchnorm=True):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(block_input)
    if (batchnorm):
        x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization(momentum=momentum)(x)
    x = Add()([block_input, x])
    return x

def normalize_z(tensor):
    import tensorflow.keras.backend as K
    t_mean = K.mean(tensor, axis=(1, 2, 3))
    t_std = K.std(tensor, axis=(1, 2, 3))
    return tf.math.divide_no_nan(tensor - t_mean[:, None, None, None], t_std[:, None, None, None])

def build_srresnet(num_filters=32, batchnorm=True, case="default", dropout_rate=0.2, normalize=False, bins=256, num_inputs=1, num_outputs=1):
    inputs = []
    for i in range(num_inputs):
        inputs.append(Input(shape=(512, 512, 1)))
        
    inp = tf.concat(inputs, axis=3)

    x = Conv2D(8, kernel_size=3, padding='same')(inp)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_1 = LeakyReLU(0.2)(x)
    if (dropout_rate != 0.0):
        x = Dropout(dropout_rate)(x)

    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_2 = LeakyReLU(0.2)(x)
    if (dropout_rate != 0.0):
        x = Dropout(dropout_rate)(x)

    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_3 = LeakyReLU(0.2)(x)
    if (dropout_rate != 0.0):
        x = Dropout(dropout_rate)(x)

    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_4 = LeakyReLU(0.2)(x)
    if (dropout_rate != 0.0):
        x = Dropout(dropout_rate)(x)

    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_5 = LeakyReLU(0.2)(x)
    if (dropout_rate != 0.0):
        x = Dropout(dropout_rate)(x)

    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_6 = LeakyReLU(0.2)(x)
    if (dropout_rate != 0.0):
        x = Dropout(dropout_rate)(x)


    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Concatenate()([x, x_6])
    if (dropout_rate != 0.0):
        x = Dropout(dropout_rate)(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_5])
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = ReLU()(x) 
    if (dropout_rate != 0.0):
        x = Dropout(dropout_rate)(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_4])
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = ReLU()(x)
    if (dropout_rate != 0.0):
        x = Dropout(dropout_rate)(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_3])
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = ReLU()(x)
    if (dropout_rate != 0.0):
        x = Dropout(dropout_rate)(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_2])
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = ReLU()(x)
    if (dropout_rate != 0.0):
        x = Dropout(dropout_rate)(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_1])
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu')(x)

    out = Conv2D(num_outputs, kernel_size=1, padding='same')(x)

    out_list = []
    for i in range(num_outputs):
        out_i = out[:, :, :, i:(i+1)]
        if (normalize):
            out_i = Lambda(normalize_z)(out_i)
        out_list.append(out_i)


    model = Model(inputs, out_list)
    return model

class customReg(tf.keras.regularizers.Regularizer):
    def __init__(self, strength, bins):
        self.strength = strength
        self.bins = bins

    def __call__(self, x):
        vec = tf.convert_to_tensor(tf.range(-1, 1, 2 / self.bins, dtype=tf.float32))
        y = vec[None, None, None, :] -  tf.reduce_sum(x * vec[None, None, None, :], axis=3)[:, :, :, None]
        z = tf.square(y) * x
        unc_estimate = tf.math.sqrt(tf.math.reduce_sum(z, axis=3))
        reg = self.strength * tf.math.reduce_sum(z)

        return reg

def softMaxAxis(x):
    return softmax(x,axis=3)

def build_unet(num_filters=32, batchnorm=False, num_inputs=1, num_outputs=1):
    inputs = []
    for i in range(num_inputs):
        inputs.append(Input(shape=(512, 512, 1)))
        
    inp = tf.concat(inputs, axis=3)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(inp)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = x_1 = PReLU(shared_axes=[1, 2])(x)
    x_1 = x

    x = Conv2D(num_filters * 2, strides=(2, 2), kernel_size=3, padding='same')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = x_2 = PReLU(shared_axes=[1, 2])(x)
    x_2 = x

    x = Conv2D(num_filters * 4, strides=(2, 2), kernel_size=3, padding='same')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = x_3 = PReLU(shared_axes=[1, 2])(x)
    x_3 = x

    x = Conv2D(num_filters * 8, strides=(2, 2), kernel_size=3, padding='same')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = x_4 = PReLU(shared_axes=[1, 2])(x)
    x_4 = x

    x = Conv2D(num_filters * 8, strides=(2, 2), kernel_size=3, padding='same')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = x_5 = PReLU(shared_axes=[1, 2])(x)
    x_5 = x

    x = Conv2D(num_filters * 8, strides=(2, 2), kernel_size=3, padding='same')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same')(x)
    if (batchnorm):
        x = BatchNormalization()(x)


    x = Conv2DTranspose(num_filters * 8, strides=(2, 2), kernel_size=2, padding='same')(x)
    x = Concatenate()([x, x_5])
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = ReLU()(x)

    x = Conv2DTranspose(num_filters * 8, strides=(2, 2), kernel_size=2, padding='same')(x)
    x = Concatenate()([x, x_4])
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = ReLU()(x)

    x = Conv2DTranspose(num_filters * 4, strides=(2, 2), kernel_size=2, padding='same')(x)
    x = Concatenate()([x, x_3])
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = ReLU()(x)

    x = Conv2DTranspose(num_filters * 2, strides=(2, 2), kernel_size=2, padding='same')(x)
    x = Concatenate()([x, x_2])
    if (batchnorm):
        x = BatchNormalization()(x)
    # x = ReLU()(x)

    x = Conv2DTranspose(num_filters, strides=(2, 2), kernel_size=2, padding='same')(x)
    x = Concatenate()([x, x_1])
    x = Conv2D(num_outputs, kernel_size=1, padding='same', activation=relu_range)(x)

    out_list = []
    for i in range(num_outputs):
        output = Lambda(normalize_z)(x[:, :, :, i:(i+1)])
        out_list.append(output)


    return Model(inputs, out_list)


def d_block(inp, fil, p = True):

    res = Conv2D(fil, 1, kernel_initializer = 'he_uniform')(inp)

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(inp)
    out = LeakyReLU(0.2)(out)
    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(out)
    out = LeakyReLU(0.2)(out)

    out = add([res, out])

    if p:
        out = AveragePooling2D()(out)

    return out

def build_discriminator(df, img_rows, img_cols, channels): 
    # import tensorflow_addons as tfa
    import tensorflow
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense, Dropout, Embedding, Reshape, Add, Lambda
    from tensorflow.keras.layers import Multiply, GaussianNoise, ReLU, LeakyReLU, Concatenate, MaxPooling2D, concatenate,  LeakyReLU, multiply
    from tensorflow.keras.models import Model
    from tensorflow.keras.initializers import RandomNormal
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import BatchNormalization
    tensorflow.random.set_seed(113)
    cha = 24
    inp = Input(shape = [512, 512, 1])


    x = d_block(inp, 1 * cha)   #128

    x = d_block(x, 2 * cha)   #64

    x = d_block(x, 4 * cha)   #32

    x = d_block(x, 8 * cha)  #16

    x = d_block(x, 16 * cha, p = False)  #8

    x = d_block(x, 16 * cha)  #4

    x = d_block(x, 32 * cha, p = False)  #4

    x = Flatten()(x)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs = inp, outputs = x)   
    # compile model
    return model
