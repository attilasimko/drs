from __future__ import print_function
from comet_ml import Experiment
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from tensorflow import experimental
from scipy import ndimage
import colorcet as cc
import time
import cv2
from tensorflow.keras import optimizers, metrics, layers
from tensorflow.keras.backend import clear_session
import random
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from scipy.ndimage import shift
import numpy as np
np.seterr(all="ignore")
import argparse
import gc
import matplotlib.pyplot as plt
import tensorflow
from numpy.random import seed
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import sys
sys.path.append("../")
import MLTK
from MLTK.synthetic_ct.models import build_discriminator, build_srresnet, build_unet
from MLTK.contrast_transfer.models import MTNN
import matplotlib.patches as patches
from MLTK.data import DataGenerator
import pydicom
random.seed(2021)
seed(2021)

def make_mask(img, thr):
    mask = img >= thr
    mask = ndimage.binary_fill_holes(mask)
    return mask

def crop_image(img, mask, defval):
    img[~mask] = defval
    return img

def znorm(img):
    return (img - np.mean(img)) / np.std(img)

def resize(img, h, w):
    img_new = cv2.resize(img[0, :, :, 0], (h, w), interpolation=cv2.INTER_CUBIC)
    return np.reshape(img_new, (1, h, w, 1))

def signal(inp, te, tr):
    img = inp[0] * (1 - np.exp(-(tr/1000) / inp[1])) * np.exp(-(te/1000)/ inp[2])
    return img

def loss(y_true, y_pred, mask):
    y_true[y_true >= 1] = 1
    y_true[y_true <= -1] = -1
    diff = np.abs(y_true - y_pred)
    return np.average(diff, weights=mask, axis=(1, 2, 3)) # 

parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--lr", default=0.00005, help="Learning rate for generator.")
parser.add_argument("--batch_size", default=16, help="Learning rate for generator.")
parser.add_argument("--gpu", default=None, help="Learning rate for generator.")
parser.add_argument("--inarray", default="", help="Learning rate for generator.")
parser.add_argument("--outarray", default="", help="Learning rate for generator.")

args = parser.parse_args()

gpu = args.gpu
inarray = args.inarray
outarray = args.outarray
lr = float(args.lr)
batch_size = 128

data_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/multi-contrast-merged" # Path to the DICOM files of decomposed PD, T1 and T2 maps
out_path = 'trained-weights/'
model_path = out_path + 'contrast_transfer_keras.h5'

ct_model = MTNN.build_generator(None)
ct_model.load_weights(model_path)
ct_model.compile(loss=['mse'])

ct_model = Model(inputs=[ct_model.inputs[0]], 
                        outputs=[ct_model.get_layer('pd').output,
                        ct_model.get_layer('t1').output,
                        ct_model.get_layer('t2').output])

num_filters = 64
batchnorm = False
unet_filters = 12
dropout_rate = 0.2
num_res_block = 12
num_inter = 1

def get_model(case):
    if (case == "I"):
        num_inputs = 1
        num_outputs = 1
        pid = "178718"
        
        model = build_srresnet(num_filters=num_filters, batchnorm=batchnorm, case=case, dropout_rate=dropout_rate, num_inputs=num_inputs, num_outputs=num_outputs)
        model.load_weights(out_path + str(pid) + ".h5")
        model.compile(optimizer=optimizers.Adam(lr), loss=["mse"], run_eagerly=True)
    elif (case == "II"):
        num_inputs = 3
        num_outputs = 1
        pid = "147537"

        model = build_srresnet(num_filters=num_filters, batchnorm=batchnorm, case=case, dropout_rate=dropout_rate, num_inputs=num_inputs, num_outputs=num_outputs)
        model.load_weights(out_path + str(pid) + ".h5")
        model.compile(optimizer=optimizers.Adam(lr), loss=["mse"], run_eagerly=True)
    return model







case_list = ["I", "II"]
for i in range(len(case_list)):
    case = case_list[i]
    model = get_model(case)
    model.compile(optimizer=optimizers.Adam(lr), loss=["mse"], run_eagerly=True)

    save_name = str(case) + "_contrast_map"
    patience = 0
    best_loss = np.inf
    count = 0

    def UpdateLoss(te, tr):
        kloss = []
        CT_STACK = []
        PD_STACK = []
        T1_STACK = []
        T2_STACK = []

        for contrast in os.listdir(os.path.join(data_path)):
            if ((contrast == "CT") | (contrast == "PD") | (contrast == "T1") | (contrast == "T2")):
                STACK = []
                for scan_file in os.listdir(os.path.join(data_path, contrast)):
                    data = pydicom.dcmread(os.path.join(data_path, contrast, scan_file))
                    STACK.append(data)
                
                if (contrast == "CT"):
                    CT_STACK = STACK
                elif (contrast == "PD"):
                    PD_STACK = STACK
                elif (contrast == "T1"):
                    T1_STACK = STACK
                elif (contrast == "T2"):
                    T2_STACK = STACK

                if ((len(CT_STACK) > 0) & (len(PD_STACK) > 0) & (len(T1_STACK) > 0) & (len(T2_STACK) > 0)):
                    if ((len(CT_STACK) == len(PD_STACK)) & (len(T1_STACK) == len(T2_STACK))):
                        CT_STACK = sorted(CT_STACK, key=lambda s: float(s.ImagePositionPatient[2]))
                        PD_STACK = sorted(PD_STACK, key=lambda s: float(s.ImagePositionPatient[2]))
                        T1_STACK = sorted(T1_STACK, key=lambda s: float(s.ImagePositionPatient[2]))
                        T2_STACK = sorted(T2_STACK, key=lambda s: float(s.ImagePositionPatient[2]))
                        
                        for i in range(int(len(CT_STACK))):
                            ct = (CT_STACK[i].RescaleIntercept + CT_STACK[i].RescaleSlope * CT_STACK[i].pixel_array) / 1000
                            ct = np.clip(cv2.resize(ct, (512, 512)), -1, 1)
                            mask = make_mask(ct, -0.2)

                            pd =  (PD_STACK[i].RescaleIntercept + PD_STACK[i].RescaleSlope * PD_STACK[i].pixel_array)
                            t1 =  (T1_STACK[i].RescaleIntercept + T1_STACK[i].RescaleSlope * T1_STACK[i].pixel_array)
                            t2 =  (T2_STACK[i].RescaleIntercept + T2_STACK[i].RescaleSlope * T2_STACK[i].pixel_array)
                            
                            ct = np.expand_dims(np.expand_dims(ct, 0), 3)
                            pd = np.expand_dims(np.expand_dims(pd, 0), 3)
                            t1 = np.expand_dims(np.expand_dims(t1, 0), 3)
                            t2 = np.expand_dims(np.expand_dims(t2, 0), 3)
                            
                            img = signal([pd, t1, t2], te_list[te], tr_list[tr])
                            img = np.interp(img, (np.min(img), np.max(img)), (0, 1))
                            img = znorm(img)
                            inp_img = img
                            target = ct

                            if (case == "II"):
                                inp_img = resize(inp_img, 256, 256)
                                inp_img = inp_img - np.min(inp_img)
                                inp_img = inp_img / np.max(inp_img)
                                inp_img = ct_model.predict_on_batch(inp_img)
                                inp_img[0] = resize(inp_img[0], 512, 512) * np.expand_dims(np.expand_dims(mask, 0), 3)
                                inp_img[0] = znorm(inp_img[0])
                                inp_img[1] = resize(inp_img[1], 512, 512) * np.expand_dims(np.expand_dims(mask, 0), 3)
                                inp_img[1] = znorm(inp_img[1])
                                inp_img[2] = resize(inp_img[2], 512, 512) * np.expand_dims(np.expand_dims(mask, 0), 3)
                                inp_img[2] = znorm(inp_img[2])

                            pred = model.predict_on_batch(inp_img)
                            current_loss = 1000 * loss(pred, target, np.expand_dims(np.expand_dims(mask, 0), 3))
                            kloss.append(current_loss)

        kloss = np.array(kloss)
        full_loss[te, tr] = np.mean(kloss)


        my_dpi = 96
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.figsize=(500, 500)
        im = ax.imshow(full_loss, cmap=cc.cm.gouldian, interpolation='none', vmin=45, vmax=65)
        fig.colorbar(im)
        plt.title("sCT Error [HU]")
        plt.xticks(np.linspace(0, len(tr_list) - 1, num=len(tr_list)), labels=tr_list, rotation=70)
        plt.yticks(np.linspace(0, len(te_list) - 1, num=len(te_list)), labels=te_list)
        plt.gca().invert_yaxis()
        ax.set_ylabel("TE [ms]", fontsize=12)
        ax.set_xlabel("TR [ms]", fontsize=12)
        plt.tight_layout()


        plt.savefig(save_name, format='png')
        plt.close('all')


    t1w_te = 8
    t1w_tr = 400
    t2w_te = 120
    t2w_tr = 4500
    num_el_te = 23
    num_el_tr = 19

    te_list = np.linspace(np.log(t1w_te), np.log(t2w_te), num=num_el_te, dtype=float)
    tr_list = np.linspace(np.log(t1w_tr), np.log(t2w_tr), num=num_el_tr, dtype=float)
    te_diff = te_list[1] - te_list[0]
    tr_diff = tr_list[1] - tr_list[0]
    te_list = [[te_list[0] - te_diff], te_list, [te_list[-1] + te_diff]]
    te_list = [item for sublist in te_list for item in sublist]
    tr_list = [[tr_list[0] - tr_diff], tr_list, [tr_list[-1] + tr_diff], [tr_list[-1] + 2 * tr_diff], [tr_list[-1] + 4 * tr_diff], [tr_list[-1] + 6 * tr_diff], [tr_list[-1] + 8 * tr_diff]]
    tr_list = [item for sublist in tr_list for item in sublist]
    te_list = np.array(np.exp(te_list), np.int32)
    tr_list = np.array(np.exp(tr_list), np.int32)
    full_loss = np.zeros((len(te_list), len(tr_list)))
    te_list[0] = 7
    te_list[1] = 8
    te_list[23] = 120
    te_list[19] = 75
    tr_list[1] = 400
    tr_list[6] = 750

    for te_idx in range(len(te_list)):
        for tr_idx in range(len(tr_list)):
            UpdateLoss(te_idx, tr_idx)