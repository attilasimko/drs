from distutils.command.build import build
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Code suppressing TF warnings and messages.
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import argparse
import numpy as np
import random
sys.path.append("../")
import time
import gc
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from data import DataGenerator
from models import InterNetLoss, KIKI
import utils as utils
from utils import compare_vif, compare_ssim, compare_vif
import tensorflow
tensorflow.get_logger().setLevel('ERROR')
from tensorflow.keras.utils import OrderedEnqueuer
from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
from tensorflow.keras.models import Model, load_model
from tensorflow import function, TensorSpec
from tensorflow import io
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import time
import warnings
from sewar.full_ref import vifp
import SimpleITK as sitk

sys.path.append("/home/attilasimko/Documents/artefacts")
from corrupt import corrupt_image, rigid_motion
from testing import bias_image, noise_image, downsample_image

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

# Generators
in_problem = "corrupt"

gen = DataGenerator(data_path + 'testing',
                    inputs=[['clean', False, 'float32']],
                    outputs=[['corrupt', False, 'float32']],
                    batch_size=32,
                    shuffle=False)

num_workers = 1

    
inp = Input(shape=(320, 320, 1))

downsample_path = "/home/attilasimko/Documents/out/amri/evals/problems/downsample_261226.h5"
motion_path = "/home/attilasimko/Documents/out/amri/evals/problems/motion_143735.h5"
noise_path = "/home/attilasimko/Documents/out/amri/evals/problems/noise_977442.h5"
bias_path = "/home/attilasimko/Documents/out/amri/evals/problems/bias_171327.h5"
kiki_path = "/home/attilasimko/Documents/out/amri/evals/final_7529.h5"

noise = KIKI(lr=lrate, num_artefacts=1)
noise.KIKI.load_weights(noise_path)

motion = KIKI(lr=lrate, num_artefacts=1)
motion.KIKI.load_weights(motion_path)

downsample = KIKI(lr=lrate, num_artefacts=1)
downsample.KIKI.load_weights(downsample_path)

bias = KIKI(lr=lrate, num_artefacts=1)
bias.KIKI.load_weights(bias_path)

kiki = KIKI(lr=lrate)
kiki.KIKI.load_weights(kiki_path)

loss_ssim = []
loss_vif = []
loss_kiki_ssim = []
loss_kiki_vif = []

loss_motion_ssim = []
loss_motion_vif = []
loss_noise_ssim = []
loss_noise_vif = []
loss_downsample_ssim = []
loss_downsample_vif = []
loss_bias_ssim = []
loss_bias_vif = []

loss_motion_downsample_ssim = []
loss_motion_downsample_vif = []
loss_noise_bias_ssim = []
loss_noise_bias_vif = []
loss_downsample_motion_ssim = []
loss_downsample_motion_vif = []
loss_bias_noise_ssim = []
loss_bias_noise_vif = []
for i in range(int(len(gen) / 10)):
    hr, lr =gen[i]
    clean = hr[0][0,:,:,0]

    if (i % 2 == 0):
        corrupt = corrupt_image(corrupt_image(clean, "motion"), "downsample")
    else:
        corrupt = corrupt_image(corrupt_image(clean, "downsample"), "motion")

    corrupt = np.expand_dims(np.expand_dims(corrupt, 0), 3)
    clean = np.expand_dims(np.expand_dims(clean, 0), 3)

    
    downsample_correction = downsample.KIKI.predict_on_batch(corrupt)
    
    motion_correction = motion.KIKI.predict_on_batch(corrupt)
    
    downsample_motion_correction = motion.KIKI.predict_on_batch(downsample_correction)
    
    motion_downsample_correction = downsample.KIKI.predict_on_batch(motion_correction)
    
    kiki_correction = kiki.KIKI.predict_on_batch(corrupt)
    
    loss_ssim.append(compare_ssim(clean, corrupt))
    loss_vif.append(compare_vif(clean, corrupt))

    loss_downsample_ssim.append(compare_ssim(clean, downsample_correction))
    loss_downsample_vif.append(compare_vif(clean, downsample_correction))

    loss_motion_ssim.append(compare_ssim(clean, motion_correction))
    loss_motion_vif.append(compare_vif(clean, motion_correction))

    loss_downsample_motion_ssim.append(compare_ssim(clean, downsample_motion_correction))
    loss_downsample_motion_vif.append(compare_vif(clean, downsample_motion_correction))

    loss_motion_downsample_ssim.append(compare_ssim(clean, motion_downsample_correction))
    loss_motion_downsample_vif.append(compare_vif(clean, motion_downsample_correction))

    loss_kiki_ssim.append(compare_ssim(clean, kiki_correction))
    loss_kiki_vif.append(compare_vif(clean, kiki_correction))

data_ssim = np.array([loss_ssim, loss_downsample_ssim, loss_downsample_motion_ssim, loss_motion_ssim, loss_motion_downsample_ssim, loss_kiki_ssim])[:, :, 0]
posthoc_ssim = sp.posthoc_nemenyi_friedman(data_ssim.T)
print(posthoc_ssim)

data_vif = np.array([loss_vif, loss_downsample_vif, loss_downsample_motion_vif, loss_motion_vif, loss_motion_downsample_vif, loss_kiki_vif])[:, :, 0]
posthoc_vif = sp.posthoc_nemenyi_friedman(data_vif.T)
print(posthoc_vif)

print(f"Baseline --- Mean Loss:\t{str(round(np.mean(loss_ssim), 4))}+-{str(round(np.std(loss_ssim), 4))}\t{str(round(np.mean(loss_vif), 4))}+-{str(round(np.std(loss_vif), 4))}")
print(f"Downsample --- Mean Loss:\t{str(round(np.mean(loss_downsample_ssim), 4))}+-{str(round(np.std(loss_downsample_ssim), 4))}\t{str(round(np.mean(loss_downsample_vif), 4))}+-{str(round(np.std(loss_downsample_vif), 4))}")
print(f"Downsample + Motion --- Mean Loss:\t{str(round(np.mean(loss_downsample_motion_ssim), 4))}+-{str(round(np.std(loss_downsample_motion_ssim), 4))}\t{str(round(np.mean(loss_downsample_motion_vif), 4))}+-{str(round(np.std(loss_downsample_motion_vif), 4))}")
print(f"Motion --- Mean Loss:\t{str(round(np.mean(loss_motion_ssim), 4))}+-{str(round(np.std(loss_motion_ssim), 4))}\t{str(round(np.mean(loss_motion_vif), 4))}+-{str(round(np.std(loss_motion_vif), 4))}")
print(f"Motion + Downsample --- Mean Loss:\t{str(round(np.mean(loss_motion_downsample_ssim), 4))}+-{str(round(np.std(loss_motion_downsample_ssim), 4))}\t{str(round(np.mean(loss_motion_downsample_vif), 4))}+-{str(round(np.std(loss_motion_downsample_vif), 4))}")
print(f"KIKI --- Mean Loss:\t{str(round(np.mean(loss_kiki_ssim), 4))}+-{str(round(np.std(loss_kiki_ssim), 4))}\t{str(round(np.mean(loss_kiki_vif), 4))}+-{str(round(np.std(loss_kiki_vif), 4))}")


loss_ssim = []
loss_vif = []
loss_kiki_ssim = []
loss_kiki_vif = []

for i in range(int(len(gen) / 10)):
    hr, lr =gen[i]
    clean = hr[0][0,:,:,0]

    corrupt = noise_image(bias_image(clean, 0.8), 2)
    

    corrupt = np.expand_dims(np.expand_dims(corrupt, 0), 3)
    clean = np.expand_dims(np.expand_dims(clean, 0), 3)

    bias_correction = bias.KIKI.predict_on_batch(corrupt)
    
    noise_correction = noise.KIKI.predict_on_batch(corrupt)
    
    bias_noise_correction = noise.KIKI.predict_on_batch(bias_correction)
    
    noise_bias_correction = bias.KIKI.predict_on_batch(noise_correction)
    
    kiki_correction = kiki.KIKI.predict_on_batch(corrupt)
    
    loss_ssim.append(compare_ssim(clean, corrupt))
    loss_vif.append(compare_vif(clean, corrupt))

    loss_bias_ssim.append(compare_ssim(clean, bias_correction))
    loss_bias_vif.append(compare_vif(clean, bias_correction))

    loss_noise_ssim.append(compare_ssim(clean, noise_correction))
    loss_noise_vif.append(compare_vif(clean, noise_correction))

    loss_bias_noise_ssim.append(compare_ssim(clean, bias_noise_correction))
    loss_bias_noise_vif.append(compare_vif(clean, bias_noise_correction))

    loss_noise_bias_ssim.append(compare_ssim(clean, noise_bias_correction))
    loss_noise_bias_vif.append(compare_vif(clean, noise_bias_correction))

    loss_kiki_ssim.append(compare_ssim(clean, kiki_correction))
    loss_kiki_vif.append(compare_vif(clean, kiki_correction))

data_ssim = np.array([loss_ssim, loss_bias_ssim, loss_bias_noise_ssim, loss_noise_ssim, loss_noise_bias_ssim, loss_kiki_ssim])[:, :, 0]
posthoc_ssim = sp.posthoc_nemenyi_friedman(data_ssim.T)
print(posthoc_ssim)

data_vif = np.array([loss_vif, loss_bias_vif, loss_bias_noise_vif, loss_noise_vif, loss_noise_bias_vif, loss_kiki_vif])[:, :, 0]
posthoc_vif = sp.posthoc_nemenyi_friedman(data_vif.T)
print(posthoc_vif)

print(f"Baseline --- Mean Loss:\t{str(round(np.mean(loss_ssim), 4))}+-{str(round(np.std(loss_ssim), 4))}\t{str(round(np.mean(loss_vif), 4))}+-{str(round(np.std(loss_vif), 4))}")
print(f"Bias --- Mean Loss:\t{str(round(np.mean(loss_bias_ssim), 4))}+-{str(round(np.std(loss_bias_ssim), 4))}\t{str(round(np.mean(loss_bias_vif), 4))}+-{str(round(np.std(loss_bias_vif), 4))}")
print(f"Bias + Noise --- Mean Loss:\t{str(round(np.mean(loss_bias_noise_ssim), 4))}+-{str(round(np.std(loss_bias_noise_ssim), 4))}\t{str(round(np.mean(loss_bias_noise_vif), 4))}+-{str(round(np.std(loss_bias_noise_vif), 4))}")
print(f"Noise --- Mean Loss:\t{str(round(np.mean(loss_noise_ssim), 4))}+-{str(round(np.std(loss_noise_ssim), 4))}\t{str(round(np.mean(loss_noise_vif), 4))}+-{str(round(np.std(loss_noise_vif), 4))}")
print(f"Noise + Bias --- Mean Loss:\t{str(round(np.mean(loss_noise_bias_ssim), 4))}+-{str(round(np.std(loss_noise_bias_ssim), 4))}\t{str(round(np.mean(loss_noise_bias_vif), 4))}+-{str(round(np.std(loss_noise_bias_vif), 4))}")
print(f"KIKI --- Mean Loss:\t{str(round(np.mean(loss_kiki_ssim), 4))}+-{str(round(np.std(loss_kiki_ssim), 4))}\t{str(round(np.mean(loss_kiki_vif), 4))}+-{str(round(np.std(loss_kiki_vif), 4))}")