import sys
import scipy
import numpy as np
import os
from os.path import dirname, join
import random
import tensorflow
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy
from matplotlib import pyplot, cm
import pydicom
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir
from scipy.stats import norm
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
# random.seed(2019)
from tensorflow.keras.models import load_model
import sys
sys.path.append("/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e")
import MLTK
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
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
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

dataset = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/RAW/MT_104/validating/'
save_dataset = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0049_3_5/validating/'
patients = os.listdir(dataset)
patients = random.sample(patients, len(patients))

TER_name = ["T2a", "T2b", "T1a", "T1b", "PD"]

fileID = 0

for pat_dir in patients:

    im_stack = np.zeros(shape=(15, 256, 256))
    
    print('Path to the DICOM directory: {}'.format(pat_dir))

    lstFilesDCM = []  # create an empty list
    lstTER = []
    for dirName, subdirList, fileList in os.walk(os.path.join(dataset,pat_dir)):
        for filename in fileList:
            lstFilesDCM.append(os.path.join(dirName,filename))# Get ref file

    for im_name in lstFilesDCM:
        img = pydicom.read_file(im_name).pixel_array
        slice_idx = int(im_name.split()[-1])-1
        contrast_idx = TER_name.index(im_name.split()[-3])
        if contrast_idx == 0:
            im_stack[slice_idx, :, :] = img / np.max(img)

    k_space = transform_image_to_kspace(im_stack)
    k_space = k_space[3:12, 64:192, 64:192]
    im_stack_small = np.abs(transform_kspace_to_image(k_space))
    for i in range(7):
        for j in range(125):
            for k in range(125):
                small_batch = im_stack_small[i:i+3, j:j+3, k:k+3]
                if (np.mean(small_batch > 0.1) >= 0.5):
                    big_batch = im_stack[i*2:i*2+5, j*2:j*2+5, k*2:k*2+5]
                    if ((np.shape(big_batch)[0] == np.shape(big_batch)[1] == np.shape(big_batch)[2]) \
                        & (np.shape(small_batch)[0] == np.shape(small_batch)[1] == np.shape(small_batch)[2])):
                        np.savez(save_dataset + str(int(fileID)),
                                big=np.array(big_batch, dtype=np.float32),
                                small=np.array(small_batch, dtype=np.float32))
                        fileID = fileID + 1