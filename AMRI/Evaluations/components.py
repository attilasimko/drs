from locale import normalize
import os
from turtle import down
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Code suppressing TF warnings and messages.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from data import DataGenerator
from models import KIKI
import utils as utils
from utils import znorm, compare_vif, compare_mse
import tensorflow
tensorflow.get_logger().setLevel('ERROR')
from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
from tensorflow.keras.models import load_model
import warnings
from sewar.full_ref import vifp
import SimpleITK as sitk

sys.path.append("/home/attilasimko/Documents/artefacts")
from testing import bias, noise_image, downsample


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
                    inputs=[['downsample', False, 'float32'],
                            ['motion', False, 'float32'],
                            ['noise', False, 'float32'],
                            ['bias', False, 'float32']],
                    outputs=[['clean', False, 'float32']],
                    batch_size=1,
                    shuffle=False)

kiki_path = "/home/attilasimko/Documents/out/amri/evals/final_7529.h5"
inp = Input(shape=(320, 320, 1))
kiki = KIKI(lr=lrate)
kiki.KIKI.load_weights(kiki_path)

reg_zeros = np.zeros((batch_size, ))

# Extra parameters

tensorflow.random.set_seed(1)

for idx in range(5):
    loss_downsample_ssim = []
    loss_downsample_vif = []
    loss_motion_ssim = []
    loss_motion_vif = []
    loss_noise_ssim = []
    loss_noise_vif = []
    loss_bias_ssim = []
    loss_bias_vif = []
    print("Component: --- " + str(idx))
    for i in range(int(len(gen) / 10)):
        hr, lr =gen[i]
        clean = lr[0]
        downsample = hr[0]
        motion = hr[1]
        noise = hr[2]
        bias = hr[3]


        if (idx >= 1):
            downsample = kiki.K1_full.predict_on_batch(downsample)
            motion = kiki.K1_full.predict_on_batch(motion)
            noise = kiki.K1_full.predict_on_batch(noise)
            bias = kiki.K1_full.predict_on_batch(bias)
        if (idx >= 2):
            downsample = kiki.I1_full.predict_on_batch(downsample)
            motion = kiki.I1_full.predict_on_batch(motion)
            noise = kiki.I1_full.predict_on_batch(noise)
            bias = kiki.I1_full.predict_on_batch(bias)
        if (idx >= 3):
            downsample = kiki.K2_full.predict_on_batch(downsample)
            motion = kiki.K2_full.predict_on_batch(motion)
            noise = kiki.K2_full.predict_on_batch(noise)
            bias = kiki.K2_full.predict_on_batch(bias)
        if (idx >= 4):
            downsample = kiki.I2_full.predict_on_batch(downsample)
            motion = kiki.I2_full.predict_on_batch(motion)
            noise = kiki.I2_full.predict_on_batch(noise)
            bias = kiki.I2_full.predict_on_batch(bias)

        clean = np.interp(clean[0, :, :, 0], (np.min(clean), np.max(clean)), (0, 1))
        clean = np.expand_dims(np.expand_dims(znorm(clean), 0), 3)

        downsample = np.interp(downsample[0, :, :, 0], (np.min(downsample), np.max(downsample)), (0, 1))
        downsample = np.expand_dims(np.expand_dims(znorm(downsample), 0), 3)

        motion = np.interp(motion[0, :, :, 0], (np.min(motion), np.max(motion)), (0, 1))
        motion = np.expand_dims(np.expand_dims(znorm(motion), 0), 3)

        noise = np.interp(noise[0, :, :, 0], (np.min(noise), np.max(noise)), (0, 1))
        noise = np.expand_dims(np.expand_dims(znorm(noise), 0), 3)

        bias = np.interp(bias[0, :, :, 0], (np.min(bias), np.max(bias)), (0, 1))
        bias = np.expand_dims(np.expand_dims(znorm(bias), 0), 3)
        
        loss_downsample_ssim.append(compare_mse(clean, downsample))
        loss_downsample_vif.append(compare_vif(clean, downsample))

        loss_motion_ssim.append(compare_mse(clean, motion))
        loss_motion_vif.append(compare_vif(clean, motion))

        loss_noise_ssim.append(compare_mse(clean, noise))
        loss_noise_vif.append(compare_vif(clean, noise))

        loss_bias_ssim.append(compare_mse(clean, bias))
        loss_bias_vif.append(compare_vif(clean, bias))

    print(f"Downsample --- Mean Loss:\t{str(round(np.mean(loss_downsample_ssim), 4))}+-{str(round(np.std(loss_downsample_ssim), 4))}\t{str(round(np.mean(loss_downsample_vif), 4))}+-{str(round(np.std(loss_downsample_vif), 4))}")
    print(f"Motion --- Mean Loss:\t{str(round(np.mean(loss_motion_ssim), 4))}+-{str(round(np.std(loss_motion_ssim), 4))}\t{str(round(np.mean(loss_motion_vif), 4))}+-{str(round(np.std(loss_motion_vif), 4))}")
    print(f"Noise --- Mean Loss:\t{str(round(np.mean(loss_noise_ssim), 4))}+-{str(round(np.std(loss_noise_ssim), 4))}\t{str(round(np.mean(loss_noise_vif), 4))}+-{str(round(np.std(loss_noise_vif), 4))}")
    print(f"Bias --- Mean Loss:\t{str(round(np.mean(loss_bias_ssim), 4))}+-{str(round(np.std(loss_bias_ssim), 4))}\t{str(round(np.mean(loss_bias_vif), 4))}+-{str(round(np.std(loss_bias_vif), 4))}")