import gc

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import tensorflow as tf
from sewar.full_ref import uqi, vifp
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops, math_ops, nn
from tensorflow.python.util.tf_export import keras_export

# import tensorflow.signal
import gc
import logging
import time

import matplotlib
# from typeguard import typechecked
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import tensorflow as tf
# import cv2
from matplotlib import cm
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from scipy import ndimage
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
# matplotlib.use('Agg')
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, Dropout, Input,
                                     Lambda, Layer, MaxPooling2D,
                                     SpatialDropout2D, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops, math_ops, nn
from tensorflow.python.util.tf_export import keras_export


# from typeguard import typechecked

def normalize_img(tensor):
    from tensorflow.keras import backend as K
    scale = K.max(tensor, axis=(1, 2, 3))
    tensor = tensorflow.math.divide_no_nan(tensor, scale[:, None, None, None])
    return tensor

def mse(A, B):
    return (np.abs(A - B)).mean(axis=None)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


######################################################################
# 2D DCT

from scipy.fftpack import dct, idct


def dct2(y, k):
    M = y.shape[0]
    N = y.shape[1]
    a = empty([M,M],float)
    b = empty([M,M],float)

    for i in range(M):
        a[i,:] = dct(y[i,:])
    for j in range(k):
        b[:,j] = dct(a[:,j])

    return b[:k, :k]


######################################################################
# 2D inverse DCT

def idct2(b, k):
    M = b.shape[0]
    N = b.shape[1]
    a = empty([M,k],float)
    y = empty([k,k],float)

    for i in range(M):
        a[i,:] = idct(b[i,:], n=k)
    for j in range(k):
        y[:,j] = idct(a[:,j], n=k)

    return y

def radial_mask(thr):
    from skimage.draw import line
    size = 320
    mask = np.zeros((size, size), dtype=np.bool)
    for i in range(0, size + 1, thr):
        img = np.zeros((size, size), dtype=np.float32)
        nmb = i
        for j in range(2):
            if (j == 1):
                rr, cc = line(0, nmb, size, size - nmb)
            else:
                rr, cc = line(nmb, 0, size - nmb, size)

            cc = cc[rr < size]
            rr = rr[rr < size]

            rr = rr[cc < size]
            cc = cc[cc < size]
            img[rr, cc] = 1
            mask = mask | (img > 0)
    return mask


def frequency_mask(thr, horizontal, extra):
    from skimage.draw import line
    size = 320
    mask = np.zeros((size, size), dtype=np.bool)
    band = int(320 / thr)

    if (horizontal):
        mask[:, int((size - band - 2) / 2):int((size + band - 2) / 2)] = True
    else:
        mask[int((size - band - 2) / 2):int((size + band - 2) / 2), :] = True

    if (extra):
        for i in range(int(size / thr)):
            if (horizontal):
                mask[:, np.random.randint(0, 320)] = True
            else:
                mask[np.random.randint(0, 320), :] = True

            

    return mask

def normalize_z(tensor):
    import tensorflow.keras.backend as K
    t_mean = K.mean(tensor, axis=(1, 2))
    t_std = K.std(tensor, axis=(1, 2))
    return tf.math.divide_no_nan(tensor - t_mean[:, None, None, :], t_std[:, None, None, :])

def normalize_k(tensor):
    import tensorflow.keras.backend as K
    t_mean = K.max(K.abs(tensor), axis=(1, 2, 3))
    return tf.math.divide_no_nan(tensor, t_mean[:, None, None, None])


def img_to_kspace(tensor):
    from tensorflow.signal import fft2d, fftshift, ifft2d, ifftshift
    dim = (1, 2)

    inp_c = tf.dtypes.complex(tensor[:, :, :, 0], tensor[:, :, :, 1])

    kspace = ifftshift(fft2d(fftshift(inp_c, axes=dim)), axes=dim)
    # kspace = tf.expand_dims(ifft2d(inp_c), 3)
    return tf.stack((tf.math.real(kspace), tf.math.imag(kspace)), 3) 

def kspace_to_img(tensor):
    from tensorflow.keras.layers import Activation
    from tensorflow.signal import fft2d, fftshift, ifft2d, ifftshift
    dim = (1, 2)

    inp_c = tf.dtypes.complex(tensor[:, :, :, 0], tensor[:, :, :, 1])
    img = fftshift(ifft2d(ifftshift(inp_c, axes=dim)), axes=dim)
    #img = tf.expand_dims(fft2d(inp_c), 3)
    return tf.stack((tf.math.real(img), tf.math.imag(img)), 3) 



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
    # img = np.stack([np.real(img), np.imag(img)], 2)
    # img = (img - np.mean(img)) / np.std(img)
    return np.real(img)

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

    # img = np.interp(img, (np.min(img), np.max(img)), (0, 1))
    k = ifftshift(fftn(fftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k

def znorm(img):
    return (img - np.mean(img)) / np.std(img)

def compare_mse(img1, img2):
    loss = []
    for i in range(np.shape(img1)[0]):
        loss.append(np.mean(np.square(img1[i, :, :, :] - img2[i, :, :, :])))
    return loss

def compare_ssim(img1, img2):
    img2 = img2
    loss = []
    for i in range(np.shape(img1)[0]):
        loss.append(ssim(img1[i, :, :, 0], img2[i, :, :, 0]))#, data_range=np.max(img1[i, :, :, 0]) - np.min(img1[i, :, :, 0])))
    return loss

def compare_vif(img1, img2):
    loss = []
    for i in range(np.shape(img1)[0]):
        loss.append(vifp(img1[i, :, :, 0], img2[i, :, :, 0]))
    return loss

def relu_range(x):
    x = tensorflow.where(K.greater(x, 0), x, K.zeros_like(x))
    x = tensorflow.where(K.less(x, 1), x, K.ones_like(x))
    # x = tensorflow.where(K.greater(x, 0), x, K.zeros_like(x))
    # x = tensorflow.where(K.less(x, 1), x, K.ones_like(x))
    
    # mean = tensorflow.reduce_mean(x, [1, 2, 3])
    # stdev = tensorflow.math.reduce_std(x, [1, 2, 3])

    # x = x - mean[:, None, None, None]
    # x = x / stdev[:, None, None, None]
    return x

def eval_gen(model, gen, title, comet):
    mse_list = []
    vif_list = []
    for i in range(len(gen)):
        hr, lr = gen[i]
        pred = model.predict_on_batch(lr[0])
        mse_list.append(compare_mse(hr[0], pred))
        vif_list.append(compare_vif(hr[0], pred))

    print(str(np.round(np.nanmean(mse_list), 5)) + " +- " + str(np.round(np.nanstd(mse_list) / np.sqrt(len(mse_list)), 10)))
    print(str(np.round(np.nanmean(vif_list), 5)) + " +- " + str(np.round(np.nanstd(vif_list) / np.sqrt(len(vif_list)), 10)))

    comet.log_metrics({title + "_mse":round(np.nanmean(mse_list), 20),
                    title + "_vif":round(np.nanmean(vif_list), 20)})

def save_progress(lr, hr, kiki, save_path, epoch, comet, idx, is_mult=False):
    N_DIGITS = 3
    save_all = False
    kc = 4
    limit = 2e-3
    rel_limit = 0.5

    comp_K1 = kiki.comp_K1.predict_on_batch(lr)
    k1 = kiki.K1_full.predict_on_batch(lr)
    comp_I1 = kiki.comp_I1.predict_on_batch(k1)
    i1 = kiki.I1_full.predict_on_batch(k1)
    comp_K2 = kiki.comp_K2.predict_on_batch(i1)
    k2 = kiki.K2_full.predict_on_batch(i1)
    comp_I2 = kiki.comp_I2.predict_on_batch(k2)
    i2 = kiki.I2_full.predict_on_batch(k2)

    if (kiki.n_artefacts != 1):
        for i in range(len(comp_K1)):
            comp_K1[i] = np.mean(np.abs(transform_kspace_to_image(comp_K1[i][0, :, :, :])))
            comp_I1[i] = np.mean(np.abs(comp_I1[i][0, :, :, :]))
            comp_K2[i] = np.mean(np.abs(transform_kspace_to_image(comp_K2[i][0, :, :, :])))
            comp_I2[i] = np.mean(np.abs(comp_I2[i][0, :, :, :]))

        if (len(comp_K1) == 1):
            comp_K1 = f"[{str(np.round(comp_K1[0], N_DIGITS))}]"
            comp_I1 = f"[{str(np.round(comp_I1[0], N_DIGITS))}]"
            comp_K2 = f"[{str(np.round(comp_K2[0], N_DIGITS))}]"
            comp_I2 = f"[{str(np.round(comp_I2[0], N_DIGITS))}]"
        else:
            comp_K1 = f"[{str(np.round(comp_K1[0], N_DIGITS))}, {str(np.round(comp_K1[1], N_DIGITS))}, {str(np.round(comp_K1[2], N_DIGITS))}, {str(np.round(comp_K1[3], N_DIGITS))} ]"
            comp_I1 = f"[{str(np.round(comp_I1[0], N_DIGITS))}, {str(np.round(comp_I1[1], N_DIGITS))}, {str(np.round(comp_I1[2], N_DIGITS))}, {str(np.round(comp_I1[3], N_DIGITS))} ]"
            comp_K2 = f"[{str(np.round(comp_K2[0], N_DIGITS))}, {str(np.round(comp_K2[1], N_DIGITS))}, {str(np.round(comp_K2[2], N_DIGITS))}, {str(np.round(comp_K2[3], N_DIGITS))} ]"
            comp_I2 = f"[{str(np.round(comp_I2[0], N_DIGITS))}, {str(np.round(comp_I2[1], N_DIGITS))}, {str(np.round(comp_I2[2], N_DIGITS))}, {str(np.round(comp_I2[3], N_DIGITS))} ]"
    else:
            comp_K1 = f"[{str(np.round(np.mean(np.abs(transform_kspace_to_image(comp_K1[0, :, :, :]))), N_DIGITS))}]"
            comp_I1 = f"[{str(np.round(np.mean(np.abs(comp_I1[0, :, :, :])), N_DIGITS))}]"
            comp_K2 = f"[{str(np.round(np.mean(np.abs(transform_kspace_to_image(comp_K2[0, :, :, :]))), N_DIGITS))}]"
            comp_I2 = f"[{str(np.round(np.mean(np.abs(comp_I2[0, :, :, :])), N_DIGITS))}]"

    k1 = k1[0, :, :, 0]
    i1 = i1[0, :, :, 0]
    k2 = k2[0, :, :, 0]
    i2 = i2[0, :, :, 0]
    lr = lr[0][0, :, :, 0]
    hr = hr[0][0, :, :, 0]
    k_lr = np.abs(transform_image_to_kspace(lr))
    k_k1 = np.abs(transform_image_to_kspace(k1))
    k_i1 = np.abs(transform_image_to_kspace(i1))
    k_k2 = np.abs(transform_image_to_kspace(k2))
    k_i2 = np.abs(transform_image_to_kspace(i2))
    k_hr = np.abs(transform_image_to_kspace(hr))

    k_lr = k_lr / np.max(np.abs(k_lr))
    k_k1 = k_k1 / np.max(np.abs(k_k1))
    k_i1 = k_i1 / np.max(np.abs(k_i1))
    k_k2 = k_k2 / np.max(np.abs(k_k2))
    k_i2 = k_i2 / np.max(np.abs(k_i2))
    k_hr = k_hr / np.max(np.abs(k_hr))

    lr = (lr - np.mean(lr)) / np.std(lr)
    k1 = (k1 - np.mean(k1)) / np.std(k1)
    i1 = (i1 - np.mean(i1)) / np.std(i1)
    k2 = (k2 - np.mean(k2)) / np.std(k2)
    i2 = (i2 - np.mean(i2)) / np.std(i2)
    hr = (hr - np.mean(hr)) / np.std(hr)

    vmx = np.max(hr)
    vmn = np.min(hr)
    dmx_min = 0.5 if is_mult else -0.25
    dmx = 1 / 0.5 if is_mult else 0.25
    kmx_min = -0.005
    kmx = 0.005

    fig = plt.figure(figsize=(40, 20))
    ax = fig.add_subplot(3, 6, 1)
    ax.imshow(lr, interpolation='none', cmap='gist_gray', vmin=vmn, vmax=vmx)
    ax.set_title("B, D, M, N")
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 2)
    ax.imshow(k1, interpolation='none', cmap='gist_gray', vmin=vmn, vmax=vmx)
    ax.set_title(comp_K1)
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 3)
    ax.imshow(i1, interpolation='none', cmap='gist_gray', vmin=vmn, vmax=vmx)
    ax.set_title(comp_I1)
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 4)
    ax.imshow(k2, interpolation='none', cmap='gist_gray', vmin=vmn, vmax=vmx)
    ax.set_title(comp_K2)
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 5)
    ax.imshow(i2, interpolation='none', cmap='gist_gray', vmin=vmn, vmax=vmx)
    ax.set_title(comp_I2)
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 6)
    ax.imshow(hr, interpolation='none', cmap='gist_gray', vmin=vmn, vmax=vmx)
    ax.axis('off')

    ax = fig.add_subplot(3, 6, 7)
    ax.imshow(lr / hr if is_mult else lr - hr, interpolation='none', cmap='coolwarm', vmin=dmx_min, vmax=dmx)
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 8)
    ax.imshow(k1 / hr if is_mult else k1 - hr, interpolation='none', cmap='coolwarm', vmin=dmx_min, vmax=dmx)
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 9)
    ax.imshow(i1 / hr if is_mult else i1 - hr, interpolation='none', cmap='coolwarm', vmin=dmx_min, vmax=dmx)
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 10)
    ax.imshow(k2 / hr if is_mult else k2 - hr, interpolation='none', cmap='coolwarm', vmin=dmx_min, vmax=dmx)
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 11)
    ax.imshow(i2 / hr if is_mult else i2 - hr, interpolation='none', cmap='coolwarm', vmin=dmx_min, vmax=dmx)
    ax.axis('off')

    ax = fig.add_subplot(3, 6, 13)
    ax.imshow(k_lr - k_hr, interpolation='none', cmap='coolwarm', vmin=kmx_min, vmax=kmx)
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 14)
    ax.imshow(k_k1 - k_hr, interpolation='none', cmap='coolwarm', vmin=kmx_min, vmax=kmx)
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 15)
    ax.imshow(k_i1 - k_hr, interpolation='none', cmap='coolwarm', vmin=kmx_min, vmax=kmx)
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 16)
    ax.imshow(k_k2 - k_hr, interpolation='none', cmap='coolwarm', vmin=kmx_min, vmax=kmx)
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 17)
    ax.imshow(k_i2 - k_hr, interpolation='none', cmap='coolwarm', vmin=kmx_min, vmax=kmx)
    ax.axis('off')
    ax = fig.add_subplot(3, 6, 18)
    ax.imshow(k_hr, interpolation='none', cmap='gist_gray', vmin=0, vmax=kmx)
    ax.axis('off')

    # plt.savefig(save_path + "pics/" + str(idx) + ".svg")
    
    if (comet != None):
        if (save_all):
            comet.log_figure(figure=plt, figure_name=str(epoch) + "_" + str(idx), overwrite=True, step=epoch)
        else:
            comet.log_figure(figure=plt, figure_name=str(idx), overwrite=True, step=epoch)
    plt.close('all')


def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

def mse_neg(orig, pred):
    orig = tf.where(K.greater(orig, -10), orig, -10*K.ones_like(orig))
    orig = tf.where(K.less(orig, 10), orig, 10*K.ones_like(orig))
    pred = tf.where(K.greater(pred, -10), pred, -10*K.ones_like(pred))
    pred = tf.where(K.less(pred, 10), pred, 10*K.ones_like(pred))
    return -tf.reduce_mean(tf.keras.metrics.mean_squared_error(orig, pred))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
