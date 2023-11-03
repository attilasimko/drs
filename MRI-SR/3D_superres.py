import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling3D, Conv3D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import concatenate
import numpy as np
from math import log10, sqrt 
from bm3d import bm3d
import os
from tensorflow.keras.optimizers import Adam
import cv2
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from scipy.ndimage import convolve
import sys
sys.path.append("../")
from MLTK.data import DataGenerator
from numpy.fft import fftshift, ifftshift, fftn, ifftn

def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions
    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    # img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    img = img / np.max(img)
    return img


def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions
    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def define_model():
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Input, MaxPooling3D, Conv3D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D, ZeroPadding1D, LeakyReLU
    from tensorflow.keras.layers import Input, Lambda
    def kspace(x):
        return tf.signal.rfft3d(tf.complex(tf.signal.ifftshift(x[0, :, :, :, 0]), tf.keras.backend.zeros_like(x[0, :, :, :, 0])))

    filt = [32, 32, 32, 32, 32, 32]
    input_shape = (15, 256, 256, 1)
    kernel_size = (3, 3, 3)
    lr = 0.001
    paddings = tf.constant([[0, 0], [0, 1], [0, 0], [0, 0], [0, 0]])

    def out_range(x):
        # from tensorflow.keras import backend as K
        x = tf.where(tf.keras.backend.greater(x, 0), x, tf.keras.backend.zeros_like(x))
        x = tf.where(tf.keras.backend.less(x, 1), x, tf.keras.backend.ones_like(x))
        return x 

    input_1 = Input(shape=input_shape, name='input_1')
    x = Conv3D(filt[0], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(input_1)
    x = Conv3D(filt[0], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x_0 = x
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)

    x = Conv3D(filt[1], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv3D(filt[1], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x_1 = x
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)

    x = Conv3D(filt[2], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv3D(filt[2], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x_2 = x
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Conv3D(filt[3], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv3D(filt[3], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x_3 = x
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)

    x = Conv3D(filt[4], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv3D(filt[4], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x_4 = x
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)

    x = Conv3D(filt[5], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv3D(filt[5], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)


    x = UpSampling3D(size=(1, 2, 2))(x)
    x = Concatenate(axis=-1)([x,x_4])
    x = Conv3D(filt[4], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv3D(filt[4], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling3D(size=(1, 2, 2))(x)
    x = Concatenate(axis=-1)([x,x_3])
    x = Conv3D(filt[3], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv3D(filt[3], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling3D(size=(2, 2, 2))(x)
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Concatenate(axis=-1)([x,x_2])
    x = Conv3D(filt[2], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv3D(filt[2], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling3D(size=(1, 2, 2))(x)
    x = Concatenate(axis=-1)([x,x_1])
    x = Conv3D(filt[1], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv3D(filt[1], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling3D(size=(1, 2, 2))(x)
    x = Concatenate(axis=-1)([x,x_0])
    x = Conv3D(filt[0], kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv3D(1, kernel_size, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Lambda(out_range)(x)

    alpha = 1
    # k_true = Lambda(kspace)(input_1)
    # k_pred = Lambda(kspace)(x)

    model = Model(inputs=input_1, outputs=x)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))
    # model.compile(loss=['mean_squared_error', 'mean_squared_error'],loss_weights=[1, alpha], optimizer=Adam(lr=lr))
    return model




if __name__ == "__main__":
    data_path = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0049/'
    # data_path = 'C:/Users/attil/Documents/DRS/Datasets/DS0049/'
    tr_gen = DataGenerator(data_path+'training',
                            inputs=[['small', True, 'float32']],
                            outputs=[['big', True, 'float32']],
                            batch_size=1,
                            shuffle=False)
    val_gen = DataGenerator(data_path+'validating',
                            inputs=[['small', True, 'float32']],
                            outputs=[['big', True, 'float32']],
                            batch_size=5,
                            shuffle=False)
    
    slices, height, width = 8, 128, 128
    model = define_model()
    print('Super-resolution using a network with # params: ' + str(model.count_params()))

    for epoch in range(20000):
        for idx in range(len(tr_gen)):
            small, big = tr_gen[idx]
            small, big = small[0], big[0]
            kspace = transform_image_to_kspace(small)
            paddings = tf.constant([[0, 0], [3, 4], [64, 64], [64, 64], [0, 0]])
            kspace = tf.pad(kspace, paddings, mode='CONSTANT', constant_values=0)
            small_kspace = np.abs(transform_kspace_to_image(kspace))
            model.train_on_batch(small_kspace, big)

        if (epoch % 1) == 0:
            small, big = val_gen[0]
            small, big = small[0], big[0]
            kspace = transform_image_to_kspace(small)
            paddings = tf.constant([[0, 0], [3, 4], [64, 64], [64, 64], [0, 0]])
            kspace = tf.pad(kspace, paddings, mode='CONSTANT', constant_values=0)
            small_kspace = np.abs(transform_kspace_to_image(kspace))
            print(f"Epoch {epoch} results:")
            big_pred = model.predict(small_kspace)
            print("PSNR: " + str(PSNR(big_pred, big)))
            fig = plt.figure(figsize=(50, 3), dpi=100)
            plt.subplot(331)
            plt.imshow(big[0, :, :, 64, 0])
            plt.axis('off')
            plt.subplot(332)
            plt.imshow(big[0, :, :, 128, 0])
            plt.axis('off')
            plt.subplot(333)
            plt.imshow(big[0, :, :, 192, 0])
            plt.axis('off')
            plt.subplot(334)
            plt.imshow(small_kspace[0, :, :, 64, 0])
            plt.axis('off')
            plt.subplot(335)
            plt.imshow(small_kspace[0, :, :, 128, 0])
            plt.axis('off')
            plt.subplot(336)
            plt.imshow(small_kspace[0, :, :, 192, 0])
            plt.axis('off')
            plt.subplot(337)
            plt.imshow(big_pred[0, :, :, 64, 0])
            plt.axis('off')
            plt.subplot(338)
            plt.imshow(big_pred[0, :, :, 128, 0])
            plt.axis('off')
            plt.subplot(339)
            plt.imshow(big_pred[0, :, :, 192, 0])
            plt.axis('off')
            fig.tight_layout()
            plt.savefig('/home/attilasimko/Documents/out/prediction_%d.png' % (epoch))
            plt.close('all')