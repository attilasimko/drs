from locale import normalize
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Code suppressing TF warnings and messages.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import subprocess
import argparse
import bm3d
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from data import DataGenerator
from models import InterNetLoss, KIKI
from utils import znorm, compare_vif, compare_ssim
import tensorflow
tensorflow.get_logger().setLevel('ERROR')
from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
from tensorflow.keras.models import load_model
import warnings
from sewar.full_ref import vifp
import SimpleITK as sitk

sys.path.append("/home/attilasimko/Documents/artefacts")
from testing import bias_image, downsample_image, noise_image


def correct_itk_diffusion(img, iter):
    time_steps = [5, 10, 15, 20, 25]
    scale = 1
    img = np.interp(img, (img.min(), img.max()), (0, scale))
    img = img[0, :, :, 0]

    imgImage = sitk.GetImageFromArray(img)
    imgImage = sitk.Cast(imgImage, sitk.sitkFloat32)
    corrector = sitk.CurvatureAnisotropicDiffusionImageFilter()
    corrector.SetTimeStep(0.0625)
    corrector.SetNumberOfIterations(5)
    outputSlice = corrector.Execute(imgImage)
    outputSlice = sitk.GetArrayFromImage(outputSlice)
    outputSlice = np.interp(outputSlice, (outputSlice.min(), outputSlice.max()), (0, 1))

    return outputSlice

def correct_bm3d(img, iter):
    denoised_image = bm3d.bm3d(img[0, :, :, 0], sigma_psd=50/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    denoised_image = np.interp(denoised_image, (denoised_image.min(), denoised_image.max()), (0, 1))

    return denoised_image

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

noise_levels = [5, 0, -5]
iter_list = [0]
for noise_level in noise_levels:
    for iter in iter_list:
        loss_kiki_ssim = []
        loss_kiki_vif = []
        loss_noise_ssim = []
        loss_noise_vif = []
        loss_ssim = []
        loss_vif = []
        loss_diffusion_ssim = []
        loss_diffusion_vif = []
        loss_bm3d_ssim = []
        loss_bm3d_vif = []
        print("Noise level: --- " + str(noise_level))
        for i in range(int(len(gen) / 10)):
            hr, lr =gen[i]
            clean = hr[0][0,:,:,0]
            corrupt = np.array(noise_image(clean, noise_level), np.float32)

            corrupt = np.expand_dims(np.expand_dims(corrupt, 0), 3)
            clean = np.expand_dims(np.expand_dims(clean, 0), 3)

            diffusion_correction = correct_itk_diffusion(corrupt, iter)
            diffusion_correction = diffusion_correction

            bm3d_correction = correct_bm3d(corrupt, iter)
            bm3d_correction = bm3d_correction

            kiki_correction = kiki.KIKI.predict_on_batch(corrupt)
            kiki_correction = np.interp(kiki_correction[0, :, :, 0], (np.min(kiki_correction), np.max(kiki_correction)), (0, 1))

            kiki_correction = np.expand_dims(np.expand_dims(znorm(kiki_correction), 0), 3)
            diffusion_correction = np.expand_dims(np.expand_dims(znorm(diffusion_correction), 0), 3)
            bm3d_correction = np.expand_dims(np.expand_dims(znorm(bm3d_correction), 0), 3)

            corrupt = np.interp(corrupt[0, :, :, 0], (np.min(corrupt), np.max(corrupt)), (0, 1))
            corrupt = np.expand_dims(np.expand_dims(znorm(corrupt), 0), 3)

            clean = np.interp(clean[0, :, :, 0], (np.min(clean), np.max(clean)), (0, 1))
            clean = np.expand_dims(np.expand_dims(znorm(clean), 0), 3)

            loss_ssim.append(compare_ssim(clean, corrupt))
            loss_vif.append(compare_vif(clean, corrupt))

            loss_kiki_ssim.append(compare_ssim(clean, kiki_correction))
            loss_kiki_vif.append(compare_vif(clean, kiki_correction))

            loss_diffusion_ssim.append(compare_ssim(clean, diffusion_correction))
            loss_diffusion_vif.append(compare_vif(clean, diffusion_correction))

            loss_bm3d_ssim.append(compare_ssim(clean, bm3d_correction))
            loss_bm3d_vif.append(compare_vif(clean, bm3d_correction))

        data_ssim = np.array([loss_ssim, loss_kiki_ssim, loss_diffusion_ssim, loss_bm3d_ssim])[:, :, 0]
        posthoc_ssim = sp.posthoc_nemenyi_friedman(data_ssim.T)
        print(posthoc_ssim)
        

        data_vif = np.array([loss_vif, loss_kiki_vif, loss_diffusion_vif, loss_bm3d_vif])[:, :, 0]
        posthoc_vif = sp.posthoc_nemenyi_friedman(data_vif.T)
        print(posthoc_vif)

        print(f"Baseline --- Mean Loss:\t{str(round(np.mean(loss_ssim), 4))}+-{str(round(np.std(loss_ssim), 4))}\t{str(round(np.mean(loss_vif), 4))}+-{str(round(np.std(loss_vif), 4))}")
        print(f"Diffusion --- Mean Loss:\t{str(round(np.mean(loss_diffusion_ssim), 4))}+-{str(round(np.std(loss_diffusion_ssim), 4))}\t{str(round(np.mean(loss_diffusion_vif), 4))}+-{str(round(np.std(loss_diffusion_vif), 4))}")
        print(f"BM3D --- Mean Loss:\t{str(round(np.mean(loss_bm3d_ssim), 4))}+-{str(round(np.std(loss_bm3d_ssim), 4))}\t{str(round(np.mean(loss_bm3d_vif), 4))}+-{str(round(np.std(loss_bm3d_vif), 4))}")
        print(f"KIKI --- Mean Loss:\t{str(round(np.mean(loss_kiki_ssim), 4))}+-{str(round(np.std(loss_kiki_ssim), 4))}\t{str(round(np.mean(loss_kiki_vif), 4))}+-{str(round(np.std(loss_kiki_vif), 4))}")
        



