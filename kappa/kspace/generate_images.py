from PIL import Image
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn
import matplotlib.pyplot as plt

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

def cartesian_mask(factor_PE, factor_SE):
    size = 256
    mask = np.zeros((size, size), dtype=bool)
    num_cols = size
       
    mask[round((num_cols - (factor_PE * size) - 1) / 2):round((num_cols + (factor_PE * size) - 1) / 2), round((num_cols - (factor_SE * size) - 1) / 2):round((num_cols + (factor_SE * size) - 1) / 2)] = True
    
    return mask

img = Image.open("kspace/input.png").convert('L')
image = np.array(img.getdata())
image = np.reshape(image, (256, 256))
acc_factor = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

save_I = 0
for i in acc_factor:
    save_J = 0
    for j in acc_factor:
        
        kspace = transform_image_to_kspace(image)
            
        mask = cartesian_mask(i, j)
        kspace = np.where(mask, kspace, (0 + 0j))
        newimage = transform_kspace_to_image(kspace)
        
        plotimg = np.zeros((256, 513))
        plotimg[0:256, 0:256] = np.log10(np.abs(kspace) + 0.0001) * 0.03
        plotimg[0:256, 257:513] = newimage
        plotimg -= np.min(plotimg)
        plotimg = ((plotimg / np.max(plotimg)) * 255).astype(np.uint8)
        im = Image.fromarray(plotimg)
        im = im.convert("RGB")
        im.save(f"kspace/kspace_{save_I}_{save_J}.png")
        save_J += 1
    save_I += 1