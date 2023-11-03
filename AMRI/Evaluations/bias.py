from locale import normalize
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Code suppressing TF warnings and messages.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
sys.path.append("../")
sys.path.append("/home/attilasimko/Documents/artefacts")
from testing import bias_image
import scikit_posthocs as sp
from data import DataGenerator
from models import InterNetLoss, KIKI
from numpy.fft import fftshift, ifftshift, fftn, ifftn, fft2, ifft2
import utils as utils
from utils import znorm, compare_vif, compare_ssim
import tensorflow
tensorflow.get_logger().setLevel('ERROR')
from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
from tensorflow.keras.models import load_model
import warnings
from sewar.full_ref import vifp
import SimpleITK as sitk


def correct_itk(img):
    controlpoints = 4
    scale = 1
    img = np.interp(img, (img.min(), img.max()), (0, scale))
    img = img[0, :, :, 0]

    imgImage = sitk.GetImageFromArray(img)
    imgImage = sitk.Cast(imgImage, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetNumberOfControlPoints(controlpoints)
    outputSlice = corrector.Execute(imgImage)
    outputSlice = sitk.GetArrayFromImage(outputSlice)

    return outputSlice

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

implicit = load_model("/home/attilasimko/Documents/bfc/bfc/weights/IN.h5", compile=False)
implicit.compile(loss=["mse"])

reg_zeros = np.zeros((batch_size, ))

bias_levels = [0.4, 0.8, 1.2]
for bias_level in bias_levels:
    loss_ssim = []
    loss_kiki_ssim = []
    loss_kiki_vif = []
    loss_imp_ssim = []
    loss_vif = []
    loss_imp_vif = []
    loss_itk_ssim = []
    loss_itk_vif = []
    print("Bias level: --- " + str(bias_level))
    for i in range(int(len(gen) / 10)):
        hr, lr = gen[i]
        clean = hr[0][0,:,:,0]
        corrupt = np.array(bias_image(clean, bias_level), np.float32)

        clean_in = cv2.resize(clean, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        corrupt_in = cv2.resize(corrupt, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        clean_in = np.interp(clean_in, (clean_in.min(), clean_in.max()), (0, 1))
        corrupt_in = np.interp(corrupt_in, (corrupt_in.min(), corrupt_in.max()), (0, 1))

        corrupt = np.expand_dims(np.expand_dims(corrupt, 0), 3)
        corrupt_in = np.expand_dims(np.expand_dims(corrupt_in, 0), 3)
        clean = np.expand_dims(np.expand_dims(clean, 0), 3)
        clean_in = np.expand_dims(np.expand_dims(znorm(clean_in), 0), 3)


        itk_correction = correct_itk(corrupt)
        itk_correction = itk_correction

        kiki_correction = kiki.KIKI.predict_on_batch(corrupt)
        kiki_correction = np.interp(kiki_correction[0, :, :, 0], (np.min(kiki_correction), np.max(kiki_correction)), (0, 1))

        implicit_correction = implicit.predict_on_batch(corrupt_in)
        implicit_correction = implicit_correction[0,:,:,0]
        implicit_correction = cv2.resize(corrupt_in[0, :, :, 0] / implicit_correction, (320, 320), interpolation=cv2.INTER_LANCZOS4)
        implicit_correction = np.interp(implicit_correction, (np.min(implicit_correction), np.max(implicit_correction)), (0, 1))
        implicit_correction = implicit_correction

        kiki_correction = np.expand_dims(np.expand_dims(znorm(kiki_correction), 0), 3)
        implicit_correction = np.expand_dims(np.expand_dims(znorm(implicit_correction), 0), 3)
        itk_correction = np.expand_dims(np.expand_dims(znorm(itk_correction), 0), 3)

        corrupt = np.interp(corrupt[0, :, :, 0], (np.min(corrupt), np.max(corrupt)), (0, 1))
        corrupt = np.expand_dims(np.expand_dims(znorm(corrupt), 0), 3)

        clean = np.interp(clean[0, :, :, 0], (np.min(clean), np.max(clean)), (0, 1))
        clean = np.expand_dims(np.expand_dims(znorm(clean), 0), 3)
        
        loss_ssim.append(compare_ssim(clean, corrupt))
        loss_vif.append(compare_vif(clean, corrupt))

        loss_kiki_ssim.append(compare_ssim(clean, kiki_correction))
        loss_kiki_vif.append(compare_vif(clean, kiki_correction))

        loss_imp_ssim.append(compare_ssim(clean, implicit_correction))
        loss_imp_vif.append(compare_vif(clean, implicit_correction))

        loss_itk_ssim.append(compare_ssim(clean, itk_correction))
        loss_itk_vif.append(compare_vif(clean, itk_correction))

    data_ssim = np.array([loss_ssim, loss_kiki_ssim, loss_imp_ssim, loss_itk_ssim])[:, :, 0]
    posthoc_ssim = sp.posthoc_nemenyi_friedman(data_ssim.T)
    print(posthoc_ssim)
    

    data_vif = np.array([loss_vif, loss_kiki_vif, loss_imp_vif, loss_itk_vif])[:, :, 0]
    posthoc_vif = sp.posthoc_nemenyi_friedman(data_vif.T)
    print(posthoc_vif)

    print(f"Baseline --- Mean Loss:\t{str(round(np.mean(loss_ssim), 4))}+-{str(round(np.std(loss_ssim), 4))}\t{str(round(np.mean(loss_vif), 4))}+-{str(round(np.std(loss_vif), 4))}")
    print(f"KIKI --- Mean Loss:\t{str(round(np.mean(loss_kiki_ssim), 4))}+-{str(round(np.std(loss_kiki_ssim), 4))}\t{str(round(np.mean(loss_kiki_vif), 4))}+-{str(round(np.std(loss_kiki_vif), 4))}")
    print(f"IMP --- Mean Loss:\t{str(round(np.mean(loss_imp_ssim), 4))}+-{str(round(np.std(loss_imp_ssim), 4))}\t{str(round(np.mean(loss_imp_vif), 4))}+-{str(round(np.std(loss_imp_vif), 4))}")
    print(f"ITK --- Mean Loss:\t{str(round(np.mean(loss_itk_ssim), 4))}+-{str(round(np.std(loss_itk_ssim), 4))}\t{str(round(np.mean(loss_itk_vif), 4))}+-{str(round(np.std(loss_itk_vif), 4))}")


