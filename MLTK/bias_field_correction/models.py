# -*- coding: utf-8 -*-
"""
This module contains ready-to-use Keras-like and Keras-compatible models.

Created on Mon Oct  9 13:48:25 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc
import json
import six
import warnings

import h5py
import numpy as np

import tensorflow
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.python.ops.gen_nn_ops import Softmax
from MLTK.data import DataGenerator
from MLTK.utils import Timer, new_divide, MyCustomCallback, WeightNormalization
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os


__all__ = ["BaseModel",
           "ConvolutionalNetwork", "FullyConvolutionalNetwork", "UNet", "GAN"]



class ComNet(object):
    def c_variance(values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values - average)**2, weights=weights)
        return np.sqrt(variance) / average
        
    def relu_range(self, x):
        from tensorflow.keras import backend as K
        x = tensorflow.where(K.greater(x, 0), x, K.zeros_like(x))
        x = tensorflow.where(K.less(x, 1), x, K.ones_like(x))
        return x

    def __init__(self, learning_rate, batch_size, data_path, save_path, filt, do, l2reg, gauss_kernel, gauss_sigma, kernel, pid,
                 comet=False, lmbd=0.1, comparator=None, implicit = True):
        from MLTK.utils import truediv, normalize_img
        self.learning_rate = learning_rate
        # self.optim = optimizers.Adam(lr=learning_rate,
        #                              beta_1=0.5, beta_2=0.999)
        self.optim = tensorflow.keras.optimizers.Adam(lr=self.learning_rate) #, decay=0.1)
        #self.optim = optimizers.RMSprop(learning_rate=self.learning_rate)
        self.pid = pid
        self.monitor = [0]
        self.data_path = data_path
        self.save_path = save_path
        self.batch_size = batch_size
        self.lmbd = lmbd
        self.getnet = comparator
        self.implicit = implicit
        if (self.implicit):
            self.tr_gen = DataGenerator(self.data_path + 'training_s',
                                inputs=[['image1', False, 'float32'],
                                                ['image2', False, 'float32']],
                                        outputs=[['data', False, 'float32'],
                                                ['zeros', 256, 'float32'],
                                                ['zeros', 256, 'float32']],
                                        batch_size=self.batch_size,
                                        shuffle=True)     
        else:
            self.tr_gen = DataGenerator(self.data_path + 'training_s',
                                inputs=[['image1', False, 'float32']],
                                        outputs=[['bias', False, 'float32']],
                                        batch_size=self.batch_size,
                                        shuffle=True)   

        self.val_gen = DataGenerator(self.data_path + 'validating',
                                    inputs=[['image', False, 'float32']],
                                     outputs=[['data', False, 'float32'],
                                              ['tissue', False, 'uint8']],
                                     batch_size=16)
        self.test_gen = DataGenerator(self.data_path + 'testing',
                                    inputs=[['image', False, 'float32']],
                                     outputs=[['data', False, 'float32'],
                                              ['tissue', False, 'uint8']],
                                      batch_size=1)
        self.input_shape = self.tr_gen.in_dims[0][1:4]

        self.corrector = BFCUNet(filt, kernel, self.input_shape, gauss_kernel, gauss_sigma, do, l2reg) # BFCNet(filt, do, l2reg, self.input_shape)
        self.corrector.compile(optimizer=self.optim, loss=['mean_squared_error'])
        if (self.implicit):
            self.comparator = self.build_comparator()
        else:
            self.comparator = self.build_explicit()

    def build_comparator(self):
        def implicit_relation(tensor):
            from tensorflow.keras import backend as K
            a = tensor[0]
            b = tensor[1]
            c = tensorflow.math.divide_no_nan(a, b)
            scale = K.mean(c, axis=(1, 2, 3))
            return tensorflow.math.divide_no_nan(c, scale[:, None, None, None])
            
        def implicit_correction(tensor):
            from tensorflow.keras import backend as K
            a = tensor[0]
            b = tensor[1]
            c = tensorflow.math.divide_no_nan(a, b)
            scale = K.max(c, axis=(1, 2, 3))
            return tensorflow.math.divide_no_nan(c, scale[:, None, None, None])

        im1 = Input(shape=self.input_shape)
        im2 = Input(shape=self.input_shape)
        im1out1 = self.corrector(im1)
        im2out1 = self.corrector(im2)

        out1 = layers.Lambda(implicit_relation)([im1out1, im2out1])

        # Regularize Image 1
        im1n = layers.Lambda(implicit_correction)([im1, im1out1])
        reg1 = self.corrector(im1n)
        reg1 = layers.Lambda(implicit_correction)([im1n, reg1])
        reg1 = tensorflow.math.subtract(im1n, reg1)

        # Regularize Image 2
        im2n = layers.Lambda(implicit_correction)([im2, im2out1])
        reg2 = self.corrector(im2n)
        reg2 = layers.Lambda(implicit_correction)([im2n, reg2])
        reg2 = tensorflow.math.subtract(im2n, reg2)

        return Model(inputs=[im1, im2], outputs=[out1, reg1, reg2])

    def build_explicit(self):
        def implicit_relation(tensor):
            from tensorflow.keras import backend as K
            a = tensor[0]
            b = tensor[1]
            c = tensorflow.math.divide_no_nan(a, b)
            scale = K.mean(c, axis=(1, 2, 3))
            return tensorflow.math.divide_no_nan(c, scale[:, None, None, None])
            
        def implicit_correction(tensor):
            from tensorflow.keras import backend as K
            a = tensor[0]
            b = tensor[1]
            c = tensorflow.math.divide_no_nan(a, b)
            scale = K.max(c, axis=(1, 2, 3))
            return tensorflow.math.divide_no_nan(c, scale[:, None, None, None])

        im1 = Input(shape=self.input_shape)
        out1 = self.corrector(im1)

        # Regularize Image 1
        #im1n = layers.Lambda(implicit_correction)([im1, out1])
        #reg1 = self.corrector(im1n)
        #reg1 = layers.Lambda(implicit_correction)([im1n, reg1])
        #reg1 = tensorflow.math.subtract(im1n, reg1)

        return Model(inputs=[im1], outputs=[out1])

    def train(self, comet=False, train_steps=1000000,
              early_stopping=False, save_interval=0):
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        # self.comparator.compile(optimizer=self.optim,
        #                         loss={'k_hat': 'mean_squared_error',
        #                              'reg1': 'mean_squared_error',
        #                              'reg2': 'mean_squared_error'})
         #.05, 0.05])
        callbacks = [
            MyCustomCallback(self.corrector, self.val_gen, self.save_path, comet),
            EarlyStopping(monitor="monitor", patience=75, verbose=2, restore_best_weights=True, mode="max")
        ]
        # self.comparator.compile(optimizer=self.optim,
        #                         loss=['mean_squared_error',
        #                               'mean_squared_error',
        #                               'mean_squared_error'],
        #                         loss_weights=[1, 0.001, 0.001]) #0.0075
        if (self.implicit):
            self.comparator.compile(optimizer=self.optim,
                        loss=['mean_squared_error',
                        'mean_squared_error',
                        'mean_squared_error'],
                        loss_weights=[1, self.lmbd, self.lmbd]) #0.0075
        else:
            self.comparator.compile(optimizer=self.optim,
                        loss=['mean_squared_error'],
                        loss_weights=[1]) #0.0075


        hist = self.comparator.fit_generator(generator=self.tr_gen,
                                             epochs=train_steps,
                                             callbacks=callbacks,
                                             max_queue_size=4,
                                             workers=0,
                                             use_multiprocessing=False,
                                             verbose=2)
        return hist, self.corrector


def BFCUNet(filt, kernel_size, input_shape, gauss_kernel, gauss_sigma, do, l2reg):
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
    pad = np.int((kernel_size-1)/2)
    paddings = tensorflow.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
    input_1 = Input(shape=input_shape, name='input_1')
    
    x = tensorflow.pad(input_1, paddings, 'REFLECT')
    x = Conv2D(4, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = Dropout(do)(x)
    x = BatchNormalization()(x)
    x_0 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = Dropout(do)(x)
    x = BatchNormalization()(x)
    x_1 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = Dropout(do)(x)
    x = BatchNormalization()(x)
    x_2 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = Dropout(do)(x)
    x = BatchNormalization()(x)
    x_3 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = Dropout(do)(x)
    x = BatchNormalization()(x)
    x_4 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = Dropout(do)(x)
    x = BatchNormalization()(x)


    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    # x = Concatenate(axis=-1)([x,x_4])
    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'valid')(x)
    # x = tensorflow.pad(x, paddings, 'REFLECT')
    # x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'valid')(x)
    # x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    # x = Concatenate(axis=-1)([x,x_3])
    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'valid')(x)
    # x = tensorflow.pad(x, paddings, 'REFLECT')
    # x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'valid')(x)
    # x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    # x = Concatenate(axis=-1)([x,x_2])
    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'valid')(x)
    # x = tensorflow.pad(x, paddings, 'REFLECT')
    # x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'valid')(x)
    # x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    # x = Concatenate(axis=-1)([x,x_1])
    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'valid')(x)
    # x = tensorflow.pad(x, paddings, 'REFLECT')
    # x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'valid')(x)
    # x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    # x = Concatenate(axis=-1)([x,x_0, input_1])
    x = tensorflow.pad(x, paddings, 'REFLECT')
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'valid')(x)
    # x = tensorflow.pad(x, paddings, 'REFLECT')
    # x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = Conv2D(1, 1, activation = 'relu', padding = 'same')(x)
    # x = AveragePooling2D(pool_size=(16, 16))(x)
    # x = UpSampling2D(size=(16, 16), interpolation='bilinear')(x)

    #pd = x[:, :, :, 0:1]
    #t1 = x[:, :, :, 1:2]
    #t2 = x[:, :, :, 2:3]

    def relu_range(x):
        import tensorflow
        from tensorflow.keras import backend as K
        x = tensorflow.where(K.greater(x, 0.5), x, K.ones_like(x)*0.5)
        x = tensorflow.where(K.less(x, 2), x, K.ones_like(x)*2)
        return x
    
    x = Lambda(relu_range)(x)

    return Model(inputs=[input_1], outputs=[x])

def U3Net(filt, dropout_rate, input_shape):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D, Conv2DTranspose
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    inp = Input(shape=input_shape, name='input_1')
    #input_2 = Input(shape=(height, width, channels), name='input_2')
    #input_3 = Input(shape=(height, width, channels), name='input_3')
    #inputs = Concatenate(axis=-1)([input_1, input_2, input_3])
    
    def encoding(x, bn = True, drop=True):
        d0 = x
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt, 3, padding = 'valid', kernel_initializer = 'he_normal')(x)
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        if bn:
            x = BatchNormalization()(x)
        if drop:
            x = Dropout(dropout_rate)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        d1 = x

        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*2, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*2, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        if bn:
            x = BatchNormalization()(x)
        if drop:
            x = Dropout(dropout_rate)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        d2 = x

        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*4, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*4, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        if bn:
            x = BatchNormalization()(x)
        if drop:
            x = Dropout(dropout_rate)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        d3 = x

        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*8, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*8, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        if bn:
            x = BatchNormalization()(x)
        if drop:
            x = Dropout(dropout_rate)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        d4 = x
        
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*16, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*16, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        if bn:
            conv5 = BatchNormalization()(x)
        if drop:
            drop5 = Dropout(dropout_rate)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        return [d0, d1, d2, d3, d4, x]

    def decoding(features, bn=True, drop=False):
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(features[5])
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*16, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*16, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        if bn:
            x = BatchNormalization()(x)
        if drop:
            x = Dropout(dropout_rate)(x)
        x = Concatenate(axis=-1)([x, features[4]])

        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = Conv2D(filt*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
        x = Concatenate(axis=-1)([x,features[3]])
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*4, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*4, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        if bn:
            x = BatchNormalization()(x)
        if drop:
            x = Dropout(dropout_rate)(x)

        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = Conv2D(filt*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
        x = Concatenate(axis=-1)([x,features[2]])
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*2, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt*2, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        if bn:
            x = BatchNormalization()(x)
        if drop:
            x = Dropout(dropout_rate)(x)

        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = Conv2D(filt, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
        x = Concatenate(axis=-1)([x,features[1]])
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        if bn:
            x = BatchNormalization()(x)

        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = Conv2D(filt, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
        x = Concatenate(axis=-1)([x,features[0]])
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        x = tf.pad(x, paddings, 'SYMMETRIC')
        x = Conv2D(filt, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(x)
        if bn:
            x = BatchNormalization()(x)
        x = Conv2D(1, 1, padding = 'same')(x)

        return x

    features = encoding(inp)


    pd = decoding(features)
    t1 = decoding(features)
    t2 = decoding(features)

    return Model(inputs=[inp], outputs=[pd, t1, t2])

def BFCNet(filt, dropout_rate, l2reg, input_shape):
    # import tensorflow
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D, UpSampling2D, Lambda, add, Conv2DTranspose
    # from blurpool import BlurPool2D, AverageBlurPooling2D, MaxBlurPooling2D
    paddings = tensorflow.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    
    def relu_range(x):
        import tensorflow
        from tensorflow.keras import backend as K
        x = tensorflow.where(K.greater(x, 0.5), x, K.ones_like(x)*0.5)
        x = tensorflow.where(K.less(x, 2), x, K.ones_like(x)*2)
        return x

    def dct_transform(x):
        import tensorflow
        from tensorflow.keras.layers import Lambda
        from tensorflow.keras import backend as K
        x = Lambda(lambda x: tensorflow.transpose(x, [0, 2, 3, 1]))(x)
        x = Lambda(lambda x: tensorflow.signal.idct(x, n=256))(x)
        x = Lambda(lambda x: tensorflow.transpose(x, [0, 2, 3, 1]))(x)
        x = Lambda(lambda x: tensorflow.signal.idct(x, n=256))(x)
        x = Lambda(lambda x: tensorflow.transpose(x, [0, 2, 3, 1]))(x)
        return x

    inp = Input(shape=input_shape, name='input_1')
    #inputs = Concatenate(axis=-1)([input_1, input_2, input_3])
    
    
    # x = Lambda(lambda x: tensorflow.transpose(x, [0, 2, 3, 1]))(inp)
    # x = Lambda(lambda x: tensorflow.signal.dct(x, norm="ortho"))(x)
    # x = Lambda(lambda x: tensorflow.transpose(x, [0, 2, 3, 1]))(x)
    # x = Lambda(lambda x: tensorflow.signal.dct(x, norm="ortho"))(x)
    # x = Lambda(lambda x: tensorflow.transpose(x, [0, 2, 3, 1]))(x)

    strides = [2, 2, 2, 2, 2, 2] 
    filts = [filt*3, filt*6, filt*8, filt*12, filt*16, filt*16]
    
    x = Conv2D(4, 3, padding = 'same')(inp) # , activity_regularizer=l2(l2reg)
    x = Conv2D(12, 3, padding = 'same')(x)
    x = Conv2D(filts[0], 3, padding = 'same')(x)
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    y = Conv2D(filts[0], 3, strides=strides[0], padding = 'same')(x)
    y = Conv2D(filts[0], 3, padding = 'same')(y)
    x = Conv2D(filts[0], 3, strides=strides[0], padding = 'same')(x)
    x = add([x, y])
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    
    y = Conv2D(filts[1], 3, strides=strides[1], padding = 'same')(x)
    y = Conv2D(filts[1], 3, padding = 'same')(y)
    x = Conv2D(filts[1], 3, strides=strides[1], padding = 'same')(x)
    x = add([x, y])
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    
    y = Conv2D(filts[2], 3, strides=strides[2], padding = 'same')(x)
    y = Conv2D(filts[2], 3, padding = 'same')(y)
    x = Conv2D(filts[2], 3, strides=strides[2], padding = 'same')(x)
    x = add([x, y])
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    
    y = Conv2D(filts[3], 3, strides=strides[3], padding = 'same')(x)
    y = Conv2D(filts[3], 3, padding = 'same')(y)
    x = Conv2D(filts[3], 3, strides=strides[3], padding = 'same')(x)
    x = add([x, y])
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    
    y = Conv2D(filts[4], 3, strides=strides[4], padding = 'same')(x)
    y = Conv2D(filts[4], 3, padding = 'same')(y)
    x = Conv2D(filts[4], 3, strides=strides[4], padding = 'same')(x)
    x = add([x, y])
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x) 
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    
    y = Conv2D(filts[5], 3, strides=strides[5], padding = 'same')(x)
    y = Conv2D(filts[5], 3, padding = 'same')(y)
    x = Conv2D(filts[5], 3, strides=strides[5], padding = 'same')(x)
    x = add([x, y])
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    x = Conv2D(1, 1, padding = 'same')(x)
    x = Lambda(dct_transform)(x)

    x = Lambda(relu_range)(x)

    return Model(inputs=[inp], outputs=[x])

def BFCNet3D(filt, dropout_rate, l2reg, input_shape):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Input, MaxPooling2D, Conv3D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D, UpSampling2D, Lambda, add, Conv2DTranspose
    from blurpool import BlurPool2D, AverageBlurPooling2D, MaxBlurPooling2D
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    
    def relu_range(x):
        from tensorflow.keras import backend as K
        x = tensorflow.where(K.greater(x, 0.5), x, K.ones_like(x)*0.5)
        x = tensorflow.where(K.less(x, 2), x, K.ones_like(x)*2)
        return x

    def dct_transform(x):
        import tensorflow
        from tensorflow.keras.layers import Lambda
        from tensorflow.keras import backend as K
        x = Lambda(lambda x: tensorflow.signal.idct(x, n=100))(x)
        x = Lambda(lambda x: tensorflow.transpose(x, [0, 2, 3, 4, 1]))(x)
        x = Lambda(lambda x: tensorflow.signal.idct(x, n=256))(x)
        x = Lambda(lambda x: tensorflow.transpose(x, [0, 2, 3, 4, 1]))(x)
        x = Lambda(lambda x: tensorflow.signal.idct(x, n=256))(x)
        x = Lambda(lambda x: tensorflow.transpose(x, [0, 2, 3, 4, 1]))(x)
        return x


    inp = Input(shape=input_shape, name='input_1')

    strides = [2, 2, 2, 2, 2, 2]
    filts = [12, 24, 32, 48, 64, 64]
    
    x = Conv3D(4, 3, padding = 'same', activity_regularizer=l2(l2reg))(inp)
    x = Conv3D(12, 3, padding = 'same', activity_regularizer=l2(l2reg))(x)
    x = Conv3D(filts[0], 3, padding = 'same', activity_regularizer=l2(l2reg))(x)
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    y = Conv3D(filts[0], 3, strides=(2, strides[0], strides[0]), padding = 'same', activity_regularizer=l2(l2reg))(x)
    y = Conv3D(filts[0], 3, padding = 'same', activity_regularizer=l2(l2reg))(y)
    x = Conv3D(filts[0], 3, strides=(2, strides[0], strides[0]), padding = 'same', activity_regularizer=l2(l2reg))(x)
    x = add([x, y])
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    
    y = Conv3D(filts[1], 3, strides=(2, strides[1], strides[1]), padding = 'same', activity_regularizer=l2(l2reg))(x)
    y = Conv3D(filts[1], 3, padding = 'same', activity_regularizer=l2(l2reg))(y)
    x = Conv3D(filts[1], 3, strides=(2, strides[1], strides[1]), padding = 'same', activity_regularizer=l2(l2reg))(x)
    x = add([x, y])
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    
    y = Conv3D(filts[2], 3, strides=(2, strides[2], strides[2]), padding = 'same', activity_regularizer=l2(l2reg))(x)
    y = Conv3D(filts[2], 3, padding = 'same', activity_regularizer=l2(l2reg))(y)
    x = Conv3D(filts[2], 3, strides=(2, strides[2], strides[2]), padding = 'same', activity_regularizer=l2(l2reg))(x)
    x = add([x, y])
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    
    y = Conv3D(filts[3], 3, strides=(2, strides[3], strides[3]), padding = 'same', activity_regularizer=l2(l2reg))(x)
    y = Conv3D(filts[3], 3, padding = 'same', activity_regularizer=l2(l2reg))(y)
    x = Conv3D(filts[3], 3, strides=(2, strides[3], strides[3]), padding = 'same', activity_regularizer=l2(l2reg))(x)
    x = add([x, y])
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    
    y = Conv3D(filts[4], 3, strides=(2, strides[4], strides[4]), padding = 'same', activity_regularizer=l2(l2reg))(x)
    y = Conv3D(filts[4], 3, padding = 'same', activity_regularizer=l2(l2reg))(y)
    x = Conv3D(filts[4], 3, strides=(2, strides[4], strides[4]), padding = 'same', activity_regularizer=l2(l2reg))(x)
    x = add([x, y])
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x) 
    
    y = Conv3D(filts[5], 3, strides=(1, strides[5], strides[5]), padding = 'same', activity_regularizer=l2(l2reg))(x)
    y = Conv3D(filts[5], 3, padding = 'same', activity_regularizer=l2(l2reg))(y)
    x = Conv3D(filts[5], 3, strides=(1, strides[5], strides[5]), padding = 'same', activity_regularizer=l2(l2reg))(x)
    x = add([x, y])
    #x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    # x = UpSampling2D(size=(32, 32), interpolation='bilinear')(x)
    x = Conv3D(1, 1, padding = 'same')(x)
    x = Lambda(dct_transform)(x)
    
    x = Lambda(relu_range)(x)

    return Model(inputs=[inp], outputs=[x])

class KNet():   
    from numpy.fft import fftshift, ifftshift, fftn, ifftn


    @tensorflow.function
    def relu_range(self, x):
        from tensorflow.keras import backend as K
        x = tensorflow.where(K.greater_equal(x, 0), x, K.zeros_like(x))
        x = tensorflow.where(K.less_equal(x, 1), x, K.ones_like(x))
        return x

    @tensorflow.function
    def select_kspace(self, tensors):
        import tensorflow as tf
        x, k = tensors
        x = tf.where(tf.keras.backend.greater(k, 0.5), x, tf.zeros_like(x))
        return x

    # @tensorflow.function
    def build_model(self, filt, kernel_size, input_shape, reg):
        from MLTK.utils import ZeroOneRegularizer 
        import tensorflow as tf
        tf.compat.v1.disable_v2_behavior()
        from tensorflow.keras.models import Model
        from tensorflow.keras import regularizers
        from tensorflow.keras.layers import Input, Lambda, MaxPooling3D, Conv3D, BatchNormalization, Dropout, UpSampling3D, Concatenate, SpatialDropout3D
        
        inp = Input(shape=input_shape, dtype=tf.complex64, batch_size=1)
        input_real = tf.math.real(inp)
        input_imag = tf.math.imag(inp)
        inp1 = Conv3D(2, kernel_size, padding = 'same')(input_real)
        inp1 = Conv3D(filt, kernel_size, padding = 'same')(inp1)

        inp2 = Conv3D(2, kernel_size, padding = 'same')(input_imag)
        inp2 = Conv3D(filt, kernel_size, padding = 'same')(inp2)

        x = Concatenate(axis=4)([inp1, inp2])
        x = Conv3D(4, kernel_size, padding = 'same')(x)
        x = BatchNormalization()(x)
        x_0 = x
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = Conv3D(filt*2, kernel_size, padding = 'same')(x)
        x = Conv3D(filt*2, kernel_size, padding = 'same')(x)
        # x = BatchNormalization()(x)
        x_1 = x
        x = MaxPooling3D(pool_size=(1, 2, 2))(x)

        x = Conv3D(filt*4, kernel_size, padding = 'same')(x)
        x = Conv3D(filt*4, kernel_size, padding = 'same')(x)
        # x = BatchNormalization()(x)
        x_2 = x
        x = MaxPooling3D(pool_size=(1, 2, 2))(x)

        x = Conv3D(filt*8, kernel_size, padding = 'same')(x)
        x = Conv3D(filt*8, kernel_size, padding = 'same')(x)
        # x = BatchNormalization()(x)
        x_3 = x
        x = MaxPooling3D(pool_size=(3, 2, 2))(x)

        x = Conv3D(filt*16, kernel_size, padding = 'same')(x)
        x = Conv3D(filt*16, kernel_size, padding = 'same')(x)
        # x = BatchNormalization()(x)
        x_4 = x
        x = MaxPooling3D(pool_size=(1, 2, 2))(x)

        x = Conv3D(filt, kernel_size, padding = 'same')(x)
        x = Conv3D(filt*16, kernel_size, padding = 'same')(x)
        # x = BatchNormalization()(x)


        x = UpSampling3D(size=(1, 2, 2))(x)
        # x = Concatenate(axis=-1)([x,x_4])
        x = Conv3D(filt*16, kernel_size, padding = 'same')(x)
        x = Conv3D(filt*16, kernel_size, padding = 'same')(x)
        # x = BatchNormalization()(x)

        x = UpSampling3D(size=(3, 2, 2))(x)
        x = tensorflow.pad(x, tensorflow.constant([[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]]), 'SYMMETRIC')
        x = Concatenate(axis=-1)([x,x_3])
        x = Conv3D(filt*8, kernel_size, padding = 'same')(x)
        x = Conv3D(filt*8, kernel_size, padding = 'same')(x)
        # x = BatchNormalization()(x)

        x = UpSampling3D(size=(1, 2, 2))(x)
        x = tensorflow.pad(x, tensorflow.constant([[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]]), 'SYMMETRIC')
        x = Concatenate(axis=-1)([x,x_2])
        x = Conv3D(filt*4, kernel_size, padding = 'same')(x)
        x = Conv3D(filt*4, kernel_size, padding = 'same')(x)
        # x = BatchNormalization()(x)

        x = UpSampling3D(size=(1, 2, 2))(x)
        x = Concatenate(axis=-1)([x,x_1])
        x = Conv3D(filt*2, kernel_size, padding = 'same')(x)
        x = Conv3D(filt*2, kernel_size, padding = 'same')(x)
        # x = BatchNormalization()(x)

        x = UpSampling3D(size=(2, 2, 2))(x)
        x = Concatenate(axis=-1)([x,input_real, input_imag])
        x = Conv3D(filt, kernel_size, padding = 'same')(x)
        x = Conv3D(filt, kernel_size, padding = 'same')(x)

        x = Conv3D(2, 1, padding = 'same', activity_regularizer=regularizers.l1(reg))(x)

        k = tf.dtypes.complex(x[:, :, :, :, 0], x[:, :, :, :, 1])
        k = tf.keras.layers.Layer(name='k')(k)
        
        img1 = tf.signal.fftshift(k)
        img1 = tf.signal.ifft3d(img1)
        img1 = tf.signal.ifftshift(img1)
        img1 = tf.math.abs(img1)
        img1 = tf.math.divide_no_nan(img1, tf.math.reduce_max(img1))

        return Model(inputs=[inp], outputs=[img1])
    


def relu_range(x):
    from tensorflow.keras import backend as K
    x = tensorflow.where(K.greater_equal(x, -1), x, -K.ones_like(x))
    x = tensorflow.where(K.less_equal(x, 1), x, K.ones_like(x))
    return x

def sCTNet(filt, kernel_size, do):
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
    input_1 = Input(shape=(256, 256, 1), name='input_1')
    
    x = Conv2D(8, kernel_size, activation = 'relu', padding = 'same')(input_1)
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = Dropout(do)(x)
    # x = BatchNormalization()(x)
    x_0 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = Dropout(do)(x)
    # x = BatchNormalization()(x)
    x_1 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = Dropout(do)(x)
    # x = BatchNormalization()(x)
    x_2 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = Dropout(do)(x)
    # x = BatchNormalization()(x)
    x_3 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = Dropout(do)(x)
    # x = BatchNormalization()(x)
    x_4 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = Dropout(do)(x)
    # x = BatchNormalization()(x)


    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_4])
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_3])
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_2])
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_1])
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_0])
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(1, 1, padding = 'same')(x)
    x = Lambda(relu_range)(x)


    return Model(inputs=[input_1], outputs=[x])

def sCTNet2(filt, kernel_size, do):
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
    input_1 = Input(shape=(256, 256, 1), name='input_1')
    input_2 = Input(shape=(256, 256, 1), name='input_2')
    inp1 = Conv2D(4, kernel_size, activation = 'relu', padding = 'same')(input_1)
    inp2 = Conv2D(4, kernel_size, activation = 'relu', padding = 'same')(input_2)
    
    x = Concatenate(axis=-1)([inp1,inp2])
    x = Conv2D(8, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = Dropout(do)(x)
    # x = BatchNormalization()(x)
    x_0 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = Dropout(do)(x)
    # x = BatchNormalization()(x)
    x_1 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = Dropout(do)(x)
    # x = BatchNormalization()(x)
    x_2 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = Dropout(do)(x)
    # x = BatchNormalization()(x)
    x_3 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = Dropout(do)(x)
    # x = BatchNormalization()(x)
    x_4 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = Dropout(do)(x)
    # x = BatchNormalization()(x)


    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_4])
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_3])
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_2])
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_1])
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    # x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_0])
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(1, 1, padding = 'same')(x)
    x = Lambda(relu_range)(x)


    return Model(inputs=[input_1, input_2], outputs=[x])

def InterNet(filt, kernel_size, do):
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
    from tensorflow.keras.layers import add
    def relu_range(x):
        from tensorflow.keras import backend as K
        x = tensorflow.where(K.greater(x, 0), x, K.zeros_like(x))
        x = tensorflow.where(K.less(x, 1), x, K.ones_like(x))
        
        # mean = tensorflow.reduce_mean(x, [1, 2, 3])
        # stdev = tensorflow.math.reduce_std(x, [1, 2, 3])

        # x = x - mean[:, None, None, None]
        # x = x / stdev[:, None, None, None]
        return x

    input_1 = Input(shape=(320, 320, 1), name='input_1')
    
    x1 = Conv2D(4, kernel_size, activation = 'relu', padding = 'same')(input_1)
    x1 = Conv2D(filt, kernel_size, activation = 'relu', padding = 'same')(x1)
    # x = Dropout(do)(x)
    x_0 = x1
    x_0 = BatchNormalization()(x_0)
    x = MaxPooling2D(pool_size=(2, 2))(x_0)

    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)

    x = add([x, Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)])
    x = BatchNormalization()(x)
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x_1 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = add([x, Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)])
    x = BatchNormalization()(x)
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)

    x = add([x, Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)])
    x = BatchNormalization()(x)
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x_2 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    
    x = add([x, Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)])
    x = BatchNormalization()(x)
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)

    x = add([x, Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)])
    x = BatchNormalization()(x)
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x_3 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = add([x, Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)])
    x = BatchNormalization()(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)

    x = add([x, Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)])
    x = BatchNormalization()(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x_4 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    
    x = add([x, Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)])
    x = BatchNormalization()(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)

    x = add([x, Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)])
    x = BatchNormalization()(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)


    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = add([x,x_4]) # Concatenate(axis=-1)([x,x_2])
    x = BatchNormalization()(x)
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = add([x,x_3])
    x = BatchNormalization()(x)
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = add([x,x_2])
    x = BatchNormalization()(x)
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = add([x,x_1])
    x = BatchNormalization()(x)
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'same')(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = add([x,x_0])
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'same')(x)
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'same')(x)
    out_img = Conv2D(1, 1, padding = 'same', activation=relu_range)(x)



    return Model(inputs=[input_1], outputs=[out_img])


def InterNetLoss(case):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
    alpha = 0.001
    def relu_range(x):
        # x = tf.square(x)
        # x = tf.math.sqrt(tf.maximum(x, 1e-9))
        
        # from tensorflow.keras import backend as K
        # from tensorflow.keras import backend as K
        # mean = tf.reduce_mean(x, [1, 2, 3])
        # stdev = tf.math.reduce_std(x, [1, 2, 3])

        # x = x - mean[:, None, None, None]
        # x = x / stdev[:, None, None, None]

        # mx = tf.reduce_max(tf.abs(x), [1, 2, 3])
        # x = tensorflow.math.divide_no_nan(x, mx[:, None, None, None])
        x = tensorflow.where(K.greater(x, -1000), x, -1000 * K.ones_like(x))
        x = tensorflow.where(K.less(x, 1000), x, 1000 * K.ones_like(x))
        return x

    def reg_dec(x):
        return 0
        # alpha = 1
        # return alpha * tf.math.abs(tf.reduce_max(tf.math.abs(x)) - 1)

    def my_regularizer(x):
        return 0
        # alpha = 0.5
        # return alpha * tf.math.reduce_sum(tf.math.abs(x[1, 1]))
        # # return alpha * tf.math.reduce_sum((tf.math.floor(x) - x) * (tf.math.ceil(x) - x))
        # if np.shape(x)[0] % 2 == 1:
        #     idx = int(np.floor(np.shape(x)[0] / 2))
        #     return alpha * tf.math.reduce_sum(tf.math.abs(x[idx, idx]))
        # else:
        #     return 0
    
    def identity():
        mat = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        mat = mat + np.random.normal(0, 0.01, (3, 3))
        mat = tf.constant_initializer(mat)
        return mat
    # ide = tf.constant_initializer(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
    # init = tf.constant_initializer(np.array([[-1, -1, -1], [-1, 18, -1], [-1, -1, -1]]))
    # init = tf.constant_initializer(np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]))

    # init = tf.constant_initializer(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
    # init2 = tf.constant_initializer(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    # init3 = tf.constant_initializer(np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]))
    
    # paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    const_max = tf.keras.constraints.MaxNorm(5)
    normal_init = 'ones'
    input_1 = Input(shape=(320, 320, 1), name='input')

    if case == "sobelsq":
        init1 = tf.constant_initializer(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8)
        init2 = tf.constant_initializer(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8)
        x = tf.pad(input_1, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT", name='pad')
        out1 = Conv2D(1, 3, 
                        padding = 'valid',
                        name = 'prediction0',
                        kernel_initializer=init1,
                        use_bias=False)(x)
        x = tf.pad(input_1, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT", name='pad')  
        out2 = Conv2D(1, 3, 
                        padding = 'valid',
                        name = 'prediction1',
                        kernel_initializer=init2,
                        use_bias=False)(x)
        out = tf.math.sqrt(tf.maximum(tf.add(tf.square(out1), tf.square(out2)), 1e-9))
        return Model(inputs=[input_1], outputs=[[out]], name='kernel')
    elif case == "sl5":
        init1 = tf.constant_initializer(np.array([[-2, -2, -4, -2, -2], 
                                                  [-1, -1, -2, -1, -1], 
                                                  [0, 0, 0, 0, 0], 
                                                  [1, 1, 2, 1, 1],
                                                  [2, 2, 4, 2, 2]]) / 36)
        init2 = tf.constant_initializer(np.array([[2, 1, 0, -1, -2],
                                                  [2, 1, 0, -1, -2],
                                                  [4, 2, 0, -2, -4],
                                                  [2, 1, 0, -1, -2],
                                                  [2, 1, 0, -1, -2]]) / 36)
        init3 = tf.constant_initializer(np.array([[0, 0, -1, 0, 0],
                                                  [0, -1, -2, -1, 0],
                                                  [-1, -2, 16, -2, -1],
                                                  [0, -1, -2, -1, 0],
                                                  [0, 0, -1, 0, 0]]) / 32)
        x = tf.pad(input_1, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), "REFLECT", name='pad')
        out1 = Conv2D(1, 5, 
                        padding = 'valid',
                        name = 'prediction0',
                        activation=relu_range,
                        kernel_initializer=init1,
                        use_bias=False)(x)
        x = tf.pad(input_1, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), "REFLECT", name='pad')  
        out2 = Conv2D(1, 5, 
                        padding = 'valid',
                        name = 'prediction1',
                        activation=relu_range,
                        kernel_initializer=init2,
                        use_bias=False)(x)
        x = tf.pad(input_1, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), "REFLECT", name='pad') 
        out3 = Conv2D(1, 5, 
                        padding = 'valid',
                        name = 'prediction2',
                        activation=relu_range,
                        kernel_initializer=init3,
                        use_bias=False)(x)
        return Model(inputs=[input_1], outputs=[out1, out2, out3], name='kernel')
    elif case == "sl3":
        init1 = tf.constant_initializer(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8)
        init2 = tf.constant_initializer(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8)
        init3 = tf.constant_initializer(np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 8)
        x = tf.pad(input_1, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT", name='pad')
        out1 = Conv2D(1, 3, 
                        padding = 'valid',
                        name = 'prediction0',
                        kernel_initializer=init1,
                        use_bias=False)(x)
        x = tf.pad(input_1, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT", name='pad')  
        out2 = Conv2D(1, 3, 
                        padding = 'valid',
                        name = 'prediction1',
                        kernel_initializer=init2,
                        use_bias=False)(x)
        x = tf.pad(input_1, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT", name='pad') 
        out3 = Conv2D(1, 3, 
                        padding = 'valid',
                        name = 'prediction2',
                        kernel_initializer=init3,
                        use_bias=False)(x)
        return Model(inputs=[input_1], outputs=[out1, out2, out3], name='kernel')
    elif case == "sobel":
        init1 = tf.constant_initializer(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8)
        init2 = tf.constant_initializer(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8)
        x = tf.pad(input_1, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT", name='pad')
        out1 = Conv2D(1, 3, 
                        padding = 'valid',
                        name = 'prediction0',
                        kernel_initializer=init1,
                        use_bias=False)(x)
        x = tf.pad(input_1, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT", name='pad')  
        out2 = Conv2D(1, 3, 
                        padding = 'valid',
                        name = 'prediction1',
                        kernel_initializer=init2,
                        use_bias=False)(x)
        return Model(inputs=[input_1], outputs=[out1, out2], name='kernel')
    elif case == "baseline":
        init = tf.constant_initializer(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
        x = tf.pad(input_1, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT", name='pad')
        out1 = Conv2D(1, 3, 
                        padding = 'valid',
                        name = 'prediction0',
                        kernel_initializer=init,
                        use_bias=False)(x)
        return Model(inputs=[input_1], outputs=[[out1]], name='kernel')
    elif case == "mixture":
        x = tf.pad(input_1, tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]]), "REFLECT", name='pad')
        out1 = Conv2D(1, 2, 
                        padding = 'valid',
                        name = 'prediction0',
                        use_bias=False)(x)
        x = tf.pad(input_1, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT", name='pad')  
        out2 = Conv2D(1, 3, 
                        padding = 'valid',
                        name = 'prediction1',
                        use_bias=False)(x)
        x = tf.pad(input_1, tf.constant([[0, 0], [1, 2], [1, 2], [0, 0]]), "REFLECT", name='pad') 
        out3 = Conv2D(1, 4, 
                        padding = 'valid',
                        name = 'prediction2',
                        use_bias=False)(x)
        return Model(inputs=[input_1], outputs=[out1, out2, out3], name='kernel')
    elif case == "baseline5":
        init = tf.constant_initializer(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
        # init = tf.constant_initializer(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
        # init = tf.constant_initializer(np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
        x = tf.pad(input_1, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), "REFLECT", name='pad')
        out1 = Conv2D(1, 5, 
                        padding = 'valid',
                        name = 'prediction0',
                        kernel_initializer=init,
                        use_bias=False)(x)

        return Model(inputs=[input_1], outputs=[[out1]], name='kernel')
    elif case == "double":
        init = tf.constant_initializer(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
        # init = tf.constant_initializer(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
        # init = tf.constant_initializer(np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
        x = tf.pad(input_1, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT", name='pad')
        x = Conv2D(1, 3, 
                        padding = 'valid',
                        name = 'prediction',
                        kernel_initializer=init,
                        use_bias=False)(x)
        x = tf.pad(x, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT", name='pad')
        out1 = Conv2D(1, 3, 
                        padding = 'valid',
                        name = 'prediction0',
                        kernel_initializer=init,
                        use_bias=False)(x)

        return Model(inputs=[input_1], outputs=[[out1]], name='kernel')
    else:
        raise ValueError('Case not valid.') 
    
def get_model(upscale_factor=3, channels=1, in_shape=40):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(in_shape, in_shape, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tensorflow.nn.depth_to_space(x, upscale_factor)

    return Model(inputs, outputs)

if __name__ == "__main__":
    import doctest
    doctest.testmod()