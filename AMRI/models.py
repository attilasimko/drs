import abc
import json
import six
import warnings

import h5py

import tensorflow
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.python.ops.gen_nn_ops import Softmax
from data import DataGenerator
from utils import img_to_kspace, kspace_to_img, normalize_z
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, LeakyReLU, UpSampling2D, Lambda, Add, Multiply
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import tensorflow as tf

def z_norm(img):
    
    return img


class img_reg(tensorflow.keras.regularizers.Regularizer):
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, x):
        out = tf.zeros_like(x[:, :, :, 0:2])
        for i in range(int(x.shape[3] / 2)):
            out += tf.abs(x[:, :, :, i*2:(i*2)+2])
        reg = self.strength * tf.reduce_mean(out)
        return reg

    def get_config(self):
        return {'strength': self.strength}

class k_reg(tensorflow.keras.regularizers.Regularizer):
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, x):
        out = tf.zeros_like(x[:, :, :, 0:2])
        for i in range(int(x.shape[3] / 2)):
            out += tf.abs(Lambda(kspace_to_img)(x[:, :, :, i*2:(i*2)+2]))
        reg = self.strength * tf.reduce_mean(out)

        return reg
        

    def get_config(self):
        return {'strength': self.strength}


class KIKI():
    def __init__(self,
                 img_input=Input(shape=(320, 320, 1)),
                 idx_input=Input(shape=(4)),
                 num_filters=64, 
                 lr=0.001,
                 alpha=[0.0, 0.0, 0.0, 0.0],
                 case=["baseline", "baseline", "baseline", "baseline"],
                 optimizer="rmsprop",
                 num_artefacts=4):

        self.n_artefacts = num_artefacts
        (self.K1, self.K1_full) = build_model(img_input, idx_input, num_artefacts, num_filters, True, alpha[0], case[0], optimizer, lr)
        (self.I1, self.I1_full) = build_model(img_input, idx_input, num_artefacts, num_filters, False, alpha[1], case[1], optimizer, lr)
        (self.K2, self.K2_full) = build_model(img_input, idx_input, num_artefacts, num_filters, True, alpha[2], case[2], optimizer, lr)
        (self.I2, self.I2_full) = build_model(img_input, idx_input, num_artefacts, num_filters, False, alpha[3], case[3], optimizer, lr)

        pred = self.K1_full(img_input)
        pred = self.I1_full(pred)
        pred = self.K2_full(pred)
        pred = self.I2_full(pred)
        self.KIKI = Model([img_input], pred)
        self.KIKI.compile(loss=["mse"], run_eagerly=False)

        comps_K1 = []
        comps_I1 = []
        comps_K2 = []
        comps_I2 = []
        for i in range(num_artefacts):
            comps_K1.append(self.K1.get_layer(f"full_{i}").output)
            comps_I1.append(self.I1.get_layer(f"full_{i}").output)
            comps_K2.append(self.K2.get_layer(f"full_{i}").output)
            comps_I2.append(self.I2.get_layer(f"full_{i}").output)

        self.comp_K1 = Model(self.K1.inputs[0], comps_K1)
        self.comp_I1 = Model(self.I1.inputs[0], comps_I1)
        self.comp_K2 = Model(self.K2.inputs[0], comps_K2)
        self.comp_I2 = Model(self.I2.inputs[0], comps_I2)

def build_model(inp, indexes, num_artefacts, num_filters, kspace, reg_alpha, loss_case, optimizer_type, lr):
    loss = InterNetLoss(inp, loss_case)
    if (optimizer_type == "adam"):
        optimizer = Adam(learning_rate=lr)
    if (optimizer_type == "rmsprop"):
        optimizer = RMSprop(learning_rate=lr)
    if (optimizer_type == "sgd"):
        optimizer = SGD(learning_rate=lr)
    if (optimizer_type == "decay"):
        lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=100, decay_rate=0.98, staircase=True)
        optimizer = SGD(lr_schedule)

    if (kspace):
        inp_data = Concatenate()([tf.math.real(inp), tf.zeros_like(inp)])
        inp_kspace = Lambda(img_to_kspace)(inp_data)
        features = feature_extraction_net(inp_kspace)#, reg=l2(reg_alpha))
        features = inference_net(features, num_filters)
        out_comp = reconstruction_net(features, num_artefacts, reg=k_reg(reg_alpha))#, reg=k_reg(reg_alpha))
        for i in range(num_artefacts):
            out_comp[i] = Lambda(kspace_to_img)(out_comp[i])
    else:
        inp_data = Concatenate()([tf.math.real(inp), tf.zeros_like(inp)])
        features = feature_extraction_net(inp_data)#, reg=l2(reg_alpha))
        features = inference_net(features, num_filters)
        out_comp = reconstruction_net(features, num_artefacts, reg=img_reg(reg_alpha))#, reg=img_reg(reg_alpha))
        
        
    out_specific = inp_data[:, :, :, 0:1]
    for i in range(num_artefacts):
        if ((i == 0)):
            out_specific = Add()([out_specific, indexes[:, i, None, None, None] * out_comp[i][:, :, :, 0:1]])
        elif (i != 0):
            out_specific = Add()([out_specific, indexes[:, i, None, None, None] * out_comp[i][:, :, :, 0:1]])

    out_specific = Lambda(normalize_z)(out_specific)
    model = Model([inp, indexes], out_specific)
    model.compile(loss=[loss], optimizer=optimizer, run_eagerly=False)

    out_full = inp_data[:, :, :, 0:1]
    for i in range(num_artefacts):
        if ((i == 0)):
            out_full = Add()([out_full, out_comp[i][:, :, :, 0:1]])
        elif (i != 0):
            out_full = Add()([out_full, out_comp[i][:, :, :, 0:1]])

    out_full = Lambda(normalize_z)(out_full)
    model_full = Model([inp], out_full)
    model_full.compile(loss=[loss], optimizer=optimizer, run_eagerly=False)

    return (model, model_full)

def feature_extraction_net(inp, num_filters=2, reg=None):
    x_0 = Conv2D(num_filters, kernel_size=3, padding="same", activity_regularizer=reg)(inp[:, :, :, 0:1])
    x_1 = Conv2D(num_filters, kernel_size=3, padding="same", activity_regularizer=reg)(inp[:, :, :, 1:2])
    return Concatenate()([x_0, x_1])

def reconstruction_net(inp, n_classes, reg=None):
    x = []
    for i in range(n_classes):
        layer = Conv2D(2, kernel_size=3, padding="same", activity_regularizer=reg, name=f"full_{i}")(inp)
        x.append(layer)

    return x

def inference_net(inp, num_filters=64, batchnorm=True, reg=None):
    x = Conv2D(8, kernel_size=3, padding='same', activity_regularizer=reg)(inp)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_1 = LeakyReLU(0.2)(x)

    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu', strides=(2, 2), activity_regularizer=reg)(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_2 = LeakyReLU(0.2)(x)

    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu', strides=(2, 2), activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_3 = LeakyReLU(0.2)(x)

    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu', strides=(2, 2), activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_4 = LeakyReLU(0.2)(x)

    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', strides=(2, 2), activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_5 = LeakyReLU(0.2)(x)

    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', strides=(2, 2), activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = x_6 = LeakyReLU(0.2)(x)

    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Concatenate()([x, x_6])

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_5])
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    if (batchnorm):
        x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_4])
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    if (batchnorm):
        x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_3])
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    if (batchnorm):
        x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_2])
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activation='relu', activity_regularizer=reg)(x)
    if (batchnorm):
        x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, x_1])
    x = Conv2D(num_filters, kernel_size=3, padding='same', activity_regularizer=reg)(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activity_regularizer=reg)(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', activity_regularizer=reg)(x)
    return x


def InterNetLoss(inp, case):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.losses import mean_squared_error
    from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D

    kernel_eye = tf.constant(np.expand_dims(np.expand_dims(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), 2), 2), dtype=tf.float32)
    kernel_prewitt_top3 = tf.constant(np.expand_dims(np.expand_dims(np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) / 6, 2), 2), dtype=tf.float32)
    kernel_prewitt_left3 = tf.constant(np.expand_dims(np.expand_dims(np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 6, 2), 2), dtype=tf.float32)
    kernel_sobel_top3 = tf.constant(np.expand_dims(np.expand_dims(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8, 2), 2), dtype=tf.float32)
    kernel_sobel_left3 = tf.constant(np.expand_dims(np.expand_dims(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8, 2), 2), dtype=tf.float32)
    kernel_laplace3 = tf.constant(np.expand_dims(np.expand_dims(np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 8, 2), 2), dtype=tf.float32)
    kernel_sobel_top5 = tf.constant(np.expand_dims(np.expand_dims(np.array([[-2, -2, -4, -2, -2], [-1, -1, -2, -1, -1], [0, 0, 0, 0, 0], [1, 1, 2, 1, 1], [2, 2, 4, 2, 2]]) / 36, 2), 2), dtype=tf.float32)
    kernel_sobel_left5 = tf.constant(np.expand_dims(np.expand_dims(np.array([[2, 1, 0, -1, -2], [2, 1, 0, -1, -2], [4, 2, 0, -2, -4], [2, 1, 0, -1, -2], [2, 1, 0, -1, -2]]) / 36, 2), 2), dtype=tf.float32)
    kernel_laplace5 = tf.constant(np.expand_dims(np.expand_dims(np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]) / 32, 2), 2), dtype=tf.float32)
    kernel_gauss = tf.constant(np.expand_dims(np.expand_dims(np.array([[0.075, 0.124, 0.075], [0.124, 0.204, 0.124], [0.075, 0.124, 0.075]]), 2), 2), dtype=tf.float32)

    def relu_range(x):
        x = tensorflow.where(K.greater(x, -1000), x, -1000 * K.ones_like(x))
        x = tensorflow.where(K.less(x, 1000), x, 1000 * K.ones_like(x))
        return x
    
    def my_mse(y_true, y_pred):
        #difference between true label and predicted label
        error = y_true-y_pred    
        #square of the error
        sqr_error = K.square(error)
        #mean of the square of the error
        mean_sqr_error = K.mean(sqr_error)
        #return the error
        return mean_sqr_error

    def l_base(y_true, y_pred):
        y_true_eye = tf.nn.conv2d(y_true, kernel_eye, strides=1, padding="SAME")
        y_pred_eye = tf.nn.conv2d(y_pred, kernel_eye, strides=1, padding="SAME")

        return mean_squared_error(y_true_eye, y_pred_eye)

    const_max = tf.keras.constraints.MaxNorm(5)
    normal_init = 'ones'

    if case == "baseline":
        def fn(y_true, y_pred):
            base_loss = l_base(y_true, y_pred)
            return base_loss
        return fn
    elif case == "sobelsq":
        def fn(y_true, y_pred):
            y_true_left = tf.nn.conv2d(y_true, kernel_sobel_left3, strides=1, padding="SAME")
            y_pred_left = tf.nn.conv2d(y_pred, kernel_sobel_left3, strides=1, padding="SAME")

            y_true_top = tf.nn.conv2d(y_true, kernel_sobel_top3, strides=1, padding="SAME")
            y_pred_top = tf.nn.conv2d(y_pred, kernel_sobel_top3, strides=1, padding="SAME")

            y_true_sqrt = tf.math.sqrt(tf.maximum(tf.add(tf.square(y_true_left), tf.square(y_true_top)), 1e-9))
            y_pred_sqrt = tf.math.sqrt(tf.maximum(tf.add(tf.square(y_pred_left), tf.square(y_pred_top)), 1e-9))

            loss = mean_squared_error(y_true_sqrt, y_pred_sqrt)
            return 3.239 * loss
        return fn
    elif case == "prewittsq":
        def fn(y_true, y_pred):
            y_true_left = tf.nn.conv2d(y_true, kernel_prewitt_left3, strides=1, padding="SAME")
            y_pred_left = tf.nn.conv2d(y_pred, kernel_prewitt_left3, strides=1, padding="SAME")

            y_true_top = tf.nn.conv2d(y_true, kernel_prewitt_top3, strides=1, padding="SAME")
            y_pred_top = tf.nn.conv2d(y_pred, kernel_prewitt_top3, strides=1, padding="SAME")

            y_true_sqrt = tf.math.sqrt(tf.maximum(tf.add(tf.square(y_true_left), tf.square(y_true_top)), 1e-9))
            y_pred_sqrt = tf.math.sqrt(tf.maximum(tf.add(tf.square(y_pred_left), tf.square(y_pred_top)), 1e-9))

            loss = mean_squared_error(y_true_sqrt, y_pred_sqrt)
            return 3.537 * loss
        return fn
    elif case == "prewitt":
        def fn(y_true, y_pred):
            y_true_left = tf.nn.conv2d(y_true, kernel_prewitt_left3, strides=1, padding="SAME")
            y_pred_left = tf.nn.conv2d(y_pred, kernel_prewitt_left3, strides=1, padding="SAME")

            loss_left = mean_squared_error(y_true_left, y_pred_left)

            y_true_top = tf.nn.conv2d(y_true, kernel_prewitt_top3, strides=1, padding="SAME")
            y_pred_top = tf.nn.conv2d(y_pred, kernel_prewitt_top3, strides=1, padding="SAME")

            loss_top = mean_squared_error(y_true_top, y_pred_top)
            
            return 4.095 * (loss_left + loss_top)
        return fn
    elif case == "sl5":
        def fn(y_true, y_pred):
            y_true_left = tf.nn.conv2d(y_true, kernel_sobel_left5, strides=1, padding="SAME")
            y_pred_left = tf.nn.conv2d(y_pred, kernel_sobel_left5, strides=1, padding="SAME")
            y_true_left = relu_range(y_true_left)
            y_pred_left = relu_range(y_pred_left)

            loss_left = mean_squared_error(y_true_left, y_pred_left)

            y_true_top = tf.nn.conv2d(y_true, kernel_sobel_top5, strides=1, padding="SAME")
            y_pred_top = tf.nn.conv2d(y_pred, kernel_sobel_top5, strides=1, padding="SAME")
            y_true_top = relu_range(y_true_top)
            y_pred_top = relu_range(y_pred_top)

            loss_top = mean_squared_error(y_true_top, y_pred_top)

            y_true_laplace = tf.nn.conv2d(y_true, kernel_laplace5, strides=1, padding="SAME")
            y_pred_laplace = tf.nn.conv2d(y_pred, kernel_laplace5, strides=1, padding="SAME")
            y_true_laplace = relu_range(y_true_laplace)
            y_pred_laplace = relu_range(y_pred_laplace)

            loss_laplace = mean_squared_error(y_true_laplace, y_pred_laplace)
            return 3.057 * (loss_left + loss_top + loss_laplace)
        return fn
    elif case == "sl3":
        def fn(y_true, y_pred):
            y_true_left = tf.nn.conv2d(y_true, kernel_sobel_left3, strides=1, padding="SAME")
            y_pred_left = tf.nn.conv2d(y_pred, kernel_sobel_left3, strides=1, padding="SAME")
            y_true_left = relu_range(y_true_left)
            y_pred_left = relu_range(y_pred_left)

            loss_left = mean_squared_error(y_true_left, y_pred_left)

            y_true_top = tf.nn.conv2d(y_true, kernel_sobel_top3, strides=1, padding="SAME")
            y_pred_top = tf.nn.conv2d(y_pred, kernel_sobel_top3, strides=1, padding="SAME")
            y_true_top = relu_range(y_true_top)
            y_pred_top = relu_range(y_pred_top)

            loss_top = mean_squared_error(y_true_top, y_pred_top)

            y_true_laplace = tf.nn.conv2d(y_true, kernel_laplace3, strides=1, padding="SAME")
            y_pred_laplace = tf.nn.conv2d(y_pred, kernel_laplace3, strides=1, padding="SAME")
            y_true_laplace = relu_range(y_true_laplace)
            y_pred_laplace = relu_range(y_pred_laplace)

            loss_laplace = mean_squared_error(y_true_laplace, y_pred_laplace)
            return 2.871 * (loss_left + loss_top + loss_laplace)
        return fn
    elif case == "sobel":
        def fn(y_true, y_pred):
            y_true_left = tf.nn.conv2d(y_true, kernel_sobel_left3, strides=1, padding="SAME")
            y_pred_left = tf.nn.conv2d(y_pred, kernel_sobel_left3, strides=1, padding="SAME")
            y_true_left = relu_range(y_true_left)
            y_pred_left = relu_range(y_pred_left)

            loss_left = mean_squared_error(y_true_left, y_pred_left)

            y_true_top = tf.nn.conv2d(y_true, kernel_sobel_top3, strides=1, padding="SAME")
            y_pred_top = tf.nn.conv2d(y_pred, kernel_sobel_top3, strides=1, padding="SAME")
            y_true_top = relu_range(y_true_top)
            y_pred_top = relu_range(y_pred_top)

            loss_top = mean_squared_error(y_true_top, y_pred_top)
            return 3.846 * (loss_left + loss_top)
        return fn
    elif case == "sobel5":
        def fn(y_true, y_pred):
            y_true_left = tf.nn.conv2d(y_true, kernel_sobel_left5, strides=1, padding="SAME")
            y_pred_left = tf.nn.conv2d(y_pred, kernel_sobel_left5, strides=1, padding="SAME")
            y_true_left = relu_range(y_true_left)
            y_pred_left = relu_range(y_pred_left)

            loss_left = mean_squared_error(y_true_left, y_pred_left)

            y_true_top = tf.nn.conv2d(y_true, kernel_sobel_top5, strides=1, padding="SAME")
            y_pred_top = tf.nn.conv2d(y_pred, kernel_sobel_top5, strides=1, padding="SAME")
            y_true_top = relu_range(y_true_top)
            y_pred_top = relu_range(y_pred_top)

            loss_top = mean_squared_error(y_true_top, y_pred_top)
            return 4.153 * (loss_left + loss_top)
        return fn
    elif case == "laplace":
        def fn(y_true, y_pred):
            y_true_laplace = tf.nn.conv2d(y_true, kernel_laplace3, strides=1, padding="SAME")
            y_pred_laplace = tf.nn.conv2d(y_pred, kernel_laplace3, strides=1, padding="SAME")
            y_true_laplace = relu_range(y_true_laplace)
            y_pred_laplace = relu_range(y_pred_laplace)

            loss_laplace = mean_squared_error(y_true_laplace, y_pred_laplace)
            return 11.313 * loss_laplace
        return fn
    elif case == "gauss":
        def fn(y_true, y_pred):
            y_true_gauss = tf.nn.conv2d(y_true, kernel_gauss, strides=1, padding="SAME")
            y_pred_gauss = tf.nn.conv2d(y_pred, kernel_gauss, strides=1, padding="SAME")

            y_true_sharp = tf.subtract(y_true, y_true_gauss)
            y_pred_sharp = tf.subtract(y_pred, y_pred_gauss)

            y_true_sharp = relu_range(y_true_sharp)
            y_pred_sharp = relu_range(y_pred_sharp)

            loss = mean_squared_error(y_true_sharp, y_pred_sharp)
            return 4.640 * loss
        return fn
    else:
        raise ValueError('Case not valid.') 

if __name__ == "__main__":
    import doctest
    doctest.testmod()