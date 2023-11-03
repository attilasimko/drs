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
                            batch_size=1,
                            shuffle=False)

    small, big = tr_gen[0]
    small, big = small[0], big[0]

    x = transform_image_to_kspace(small)
    paddings = tf.constant([[0, 0], [3, 4], [64, 64], [64, 64], [0, 0]])
    x = tf.pad(x, paddings, mode='CONSTANT', constant_values=0)
    x = np.abs(transform_kspace_to_image(x))
    print(np.mean(big - x))




    x = transform_image_to_kspace(big)
    x = x[:, 3:11, 64:192, 64:192, :]
    x = np.abs(transform_kspace_to_image(x))
    print(np.mean(small - x))