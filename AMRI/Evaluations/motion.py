from locale import normalize
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Code suppressing TF warnings and messages.
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import bm3d
import argparse
import numpy as np
import cv2
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
sys.path.append("../")
import scikit_posthocs as sp
from data import DataGenerator
from models import InterNetLoss, KIKI
from numpy.fft import fftshift, ifftshift, fftn, ifftn, fft2, ifft2
import utils as utils
from utils import znorm, compare_vif, compare_ssim, transform_kspace_to_image
import tensorflow
tensorflow.get_logger().setLevel('ERROR')
from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
from tensorflow.keras.models import load_model
import warnings
from sewar.full_ref import vifp
import SimpleITK as sitk

sys.path.append("/home/attilasimko/Documents/artefacts")
from corrupt import rigid_motion

def blur_edge(img, d=31):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)

def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern

def defocus_kernel(d, sz=65):
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern


warnings.filterwarnings("ignore")
np.random.seed(113)
tensorflow.get_logger().setLevel('ERROR')
pid = os.getpid()
print(pid)

parser = argparse.ArgumentParser(description='Welcome.')
# Arguments to optimize
parser.add_argument("--lr", default=0.0001) # [0.5, 0.1, 0.05]
parser.add_argument("--optimizer", default="rmsprop")
parser.add_argument("--loss", default="mean_squared_error")
parser.add_argument("--batch_size", default=4) # [0.5, 0.1, 0.05]
parser.add_argument("--alpha", default=0.001) # [0.5, 0.1, 0.05]
parser.add_argument("--kspace", default="True") # [0.5, 0.1, 0.05]
parser.add_argument("--beta", default=1.0) # [0.5, 0.1, 0.05]
parser.add_argument("--case", default="baseline") # [0, 1, 2]
parser.add_argument("--gpu", default=None)
parser.add_argument("--base", default=None)
args = parser.parse_args()

lrate = float(args.lr)
batch_size = int(args.batch_size)
kspace = args.kspace == "True"
gpu = args.gpu
case = str(args.case)
beta = float(args.beta)
alpha = float(args.alpha) * beta

# Paths
data_path = '/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/DSets/DS0059/'
save_path = "/home/attilasimko/Documents/out/amri/"
base = args.base
plt_slice = 0

gen = DataGenerator(data_path + 'testing',
                    inputs=[['clean', False, 'float32']],
                    outputs=[],
                    batch_size=1,
                    shuffle=False)

kiki_path = "/home/attilasimko/Documents/out/amri/evals/final_7529.h5"
inp = Input(shape=(320, 320, 1))
kiki = KIKI(lr=lrate)
kiki.KIKI.load_weights(kiki_path)


reg_zeros = np.zeros((batch_size, ))

# Extra parameters

tensorflow.random.set_seed(1)

weights = [0.05]
motion_levels = [0.25, 0.125, 0.04]
iter_list = [0]
for weight in weights:
    for motion_level in motion_levels:
        for iter in iter_list:
            loss_kiki_ssim = []
            loss_kiki_vif = []
            loss_motion_ssim = []
            loss_motion_vif = []
            loss_ssim = []
            loss_vif = []
            loss_opencv_ssim = []
            loss_opencv_vif = []
            print("motion level: --- " + str(motion_level))
            for i in range(int(len(gen) / 100)):
                hr, lr =gen[i]
                clean = hr[0][0,:,:,0]
                corrupt = np.array(transform_kspace_to_image(rigid_motion(clean, random.choice(["LR", "AP"]), np.random.randint(1, 9), 3, 10, motion_level)), np.float32)
                corrupt = znorm(corrupt)

                corrupt = np.expand_dims(np.expand_dims(corrupt, 0), 3)
                clean = np.expand_dims(np.expand_dims(clean, 0), 3)


                opencv_correction = denoise_tv_chambolle(corrupt[0, :, :, 0], weight=weight)
                opencv_correction = np.interp(opencv_correction, (np.min(opencv_correction), np.max(opencv_correction)), (0, 1))

                kiki_correction = kiki.KIKI.predict_on_batch(corrupt)
                kiki_correction = np.interp(kiki_correction[0, :, :, 0], (np.min(kiki_correction), np.max(kiki_correction)), (0, 1))

                kiki_correction = np.expand_dims(np.expand_dims(znorm(kiki_correction), 0), 3)
                opencv_correction = np.expand_dims(np.expand_dims(znorm(opencv_correction), 0), 3)

                corrupt = np.interp(corrupt[0, :, :, 0], (np.min(corrupt), np.max(corrupt)), (0, 1))
                corrupt = np.expand_dims(np.expand_dims(znorm(corrupt), 0), 3)

                clean = np.interp(clean[0, :, :, 0], (np.min(clean), np.max(clean)), (0, 1))
                clean = np.expand_dims(np.expand_dims(znorm(clean), 0), 3)

                loss_ssim.append(compare_ssim(clean, corrupt))
                loss_vif.append(compare_vif(clean, corrupt))

                loss_kiki_ssim.append(compare_ssim(clean, kiki_correction))
                loss_kiki_vif.append(compare_vif(clean, kiki_correction))

                loss_opencv_ssim.append(compare_ssim(clean, opencv_correction))
                loss_opencv_vif.append(compare_vif(clean, opencv_correction))

            data_ssim = np.array([loss_ssim, loss_kiki_ssim, loss_opencv_ssim])[:, :, 0]
            posthoc_ssim = sp.posthoc_nemenyi_friedman(data_ssim.T)
            print(posthoc_ssim)
            
            data_vif = np.array([loss_vif, loss_kiki_vif, loss_opencv_vif])[:, :, 0]
            posthoc_vif = sp.posthoc_nemenyi_friedman(data_vif.T)
            print(posthoc_vif)

            print(f"Baseline --- Mean Loss:\t{str(round(np.mean(loss_ssim), 4))}+-{str(round(np.std(loss_ssim), 4))}\t{str(round(np.mean(loss_vif), 4))}+-{str(round(np.std(loss_vif), 4))}")
            print(f"OpenCV --- Mean Loss:\t{str(round(np.mean(loss_opencv_ssim), 4))}+-{str(round(np.std(loss_opencv_ssim), 4))}\t{str(round(np.mean(loss_opencv_vif), 4))}+-{str(round(np.std(loss_opencv_vif), 4))}")
            print(f"KIKI --- Mean Loss:\t{str(round(np.mean(loss_kiki_ssim), 4))}+-{str(round(np.std(loss_kiki_ssim), 4))}\t{str(round(np.mean(loss_kiki_vif), 4))}+-{str(round(np.std(loss_kiki_vif), 4))}")