from distutils.log import fatal
import os
import pydicom
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil
import cv2
from tensorflow.keras.models import load_model
import sys
from MLTK.data import DataGenerator
from MLTK.synthetic_ct.models import build_srresnet
from tensorflow import function, TensorSpec
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import concatenate
from tensorflow import io
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
np.random.seed(2001)

def signal(inp, te, tr):
    return inp[0] * (1 - np.exp(-(tr/1000) / inp[1])) * np.exp(-(te/1000)/ inp[2])

def make_mask(img, thr):
    mask = img >= thr
    mask = ndimage.binary_dilation(mask, iterations=2)
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask = sizes > 512*512*0.08
    mask = mask[label_im]
    mask = ndimage.binary_fill_holes(mask)
    return mask

def crop_image(img, mask, defval):
    img[~mask] = defval
    return img

data_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/Pelvis_2.2" # Path to the DICOM images
base_dir =  '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0060/' # Output directory of pre-processed data samples
st1w = False

model_path = 'trained-weights/contrast_transfer_keras.h5'
model = load_model(model_path, compile=False)
model = Model(model.inputs[0], concatenate((model.outputs[0], model.outputs[1], model.outputs[2])))
model.compile(loss=['mse'])

sfatmodel = build_srresnet(num_filters=64, batchnorm="True", case="IV", dropout_rate=0.2, alpha=0.0, bins=0, num_inputs=4, num_outputs=2, normalize=True)
sfatmodel.load_weights(base_dir + "sfatwater.h5")
sfatmodel.compile(loss=["mse"])

oopmodel = build_srresnet(num_filters=64, batchnorm="True", case="IV", dropout_rate=0.2, alpha=0.0, bins=0, num_inputs=1, num_outputs=1, normalize=True)
oopmodel.load_weights(base_dir + "oop.h5")
oopmodel.compile(loss=["mse"])

t1wte = 7
t1wtr = 500

os.mkdir(base_dir)
os.mkdir(base_dir + "testing")
os.mkdir(base_dir + "validating")
os.mkdir(base_dir + "training")

patient_idx = 0
lst = os.listdir(data_path)
lst.sort()
np.random.shuffle(lst)
only_training = False

for patient in lst:
    save_path = ""
    mult = 1
    unpaired = False
    FAT_STACK = []
    OOP_STACK = []
    WATER_STACK = []
    MR_STACK = []
    CT_STACK = []

    if (patient_idx / len(lst) < 0.8):
        save_path =  base_dir + 'training/'
    elif (patient_idx / len(lst) < 0.9):
        save_path = base_dir + 'validating/'
    else:
        save_path = base_dir + 'testing/'

    if (not(os.path.isdir(os.path.join(data_path, patient)))):
        print("Patient does not exist (" + str(patient) + ")")
        continue

    for contrast in os.listdir(os.path.join(data_path, patient)):
        if ((contrast == "MR") | (contrast == "FAT") | (contrast == "WATER") | (contrast == "CT") | (contrast == "OOP")):
            STACK = []
            for scan_file in os.listdir(os.path.join(data_path, patient, contrast)):
                data = pydicom.dcmread(os.path.join(data_path, patient, contrast, scan_file))
                STACK.append(data)
            
            if (contrast == "MR"):
                MR_STACK = STACK
            elif (contrast == "OOP"):
                OOP_STACK = STACK
            elif (contrast == "FAT"):
                FAT_STACK = STACK
            elif (contrast == "WATER"):
                WATER_STACK = STACK
            elif (contrast == "CT"):
                CT_STACK = STACK

            if ((len(MR_STACK) > 0) & (len(CT_STACK) > 0) & (len(FAT_STACK) > 0) & (len(OOP_STACK) > 0) & (len(WATER_STACK) > 0)):
                if ((len(MR_STACK) == len(CT_STACK)) & (len(OOP_STACK) == len(FAT_STACK))):
                    CT_STACK = sorted(CT_STACK, key=lambda s: float(s.ImagePositionPatient[2]))
                    OOP_STACK = sorted(OOP_STACK, key=lambda s: float(s.ImagePositionPatient[2]))
                    MR_STACK = sorted(MR_STACK, key=lambda s: float(s.ImagePositionPatient[2]))
                    FAT_STACK = sorted(FAT_STACK, key=lambda s: float(s.ImagePositionPatient[2]))
                    WATER_STACK = sorted(WATER_STACK, key=lambda s: float(s.ImagePositionPatient[2]))
                    
                    print(str(patient) + "\t" + str(len(CT_STACK)))
                    for i in range(len(MR_STACK)):
                        ct = (CT_STACK[i].RescaleIntercept + CT_STACK[i].RescaleSlope * CT_STACK[i].pixel_array) / 1000
                        ct = np.clip(cv2.resize(ct, (512, 512)), -1, 1)
                        mask = make_mask(ct, -0.2)
                        ct = crop_image(ct, mask, -1)

                        mr =  MR_STACK[i].pixel_array / np.mean(MR_STACK[i].pixel_array)
                        mr = cv2.resize(mr, (512, 512))
                        mr = crop_image(mr, mask, 0)
                        if ((np.max(mr) == 0) | (np.isnan(mr).any())): 
                            continue
                        mr =  (mr - np.mean(mr)) / np.std(mr)

                        oop =  OOP_STACK[i].pixel_array / np.mean(OOP_STACK[i].pixel_array)
                        oop = cv2.resize(oop, (512, 512))
                        oop = crop_image(oop, mask, 0)
                        if ((np.max(oop) == 0) | (np.isnan(oop).any())): 
                            continue
                        oop =  (oop - np.mean(oop)) / np.std(oop)

                        fat =  FAT_STACK[i].pixel_array / np.mean(FAT_STACK[i].pixel_array)
                        fat = cv2.resize(fat, (512, 512))
                        fat = crop_image(fat, mask, 0)
                        fat =  (fat - np.mean(fat)) / np.std(fat)

                        water =  WATER_STACK[i].pixel_array / np.mean(WATER_STACK[i].pixel_array)
                        water = cv2.resize(water, (512, 512))
                        water = crop_image(water, mask, 0)
                        water =  (water - np.mean(water)) / np.std(water)


                        smr = cv2.resize(mr, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                        smr = smr - np.min(smr)
                        smr = smr / np.max(smr)

                        soop = oopmodel.predict_on_batch(np.expand_dims(np.expand_dims(mr, 0), 3))
                        soop = np.interp(soop, (np.min(soop), np.max(soop)), (0, 1))
                        soop = crop_image(soop[0, :, :, 0], mask, 0)
                        soop =  (soop - np.mean(soop)) / np.std(soop)

                        comps = model.predict_on_batch(np.expand_dims(np.expand_dims(smr, 0), 3))
                        pd = cv2.resize(comps[0, :, :, 0], (512, 512), interpolation=cv2.INTER_LANCZOS4)
                        pd = crop_image(pd, mask, 0)

                        t1 = cv2.resize(comps[0, :, :, 1], (512, 512), interpolation=cv2.INTER_LANCZOS4)
                        t1 = crop_image(t1, mask, 0)

                        t2 = cv2.resize(comps[0, :, :, 2], (512, 512), interpolation=cv2.INTER_LANCZOS4)
                        t2 = crop_image(t2, mask, 0)

                        comps = sfatmodel.predict_on_batch([np.expand_dims(np.expand_dims(mr, 0), 3),
                                                            np.expand_dims(np.expand_dims(pd, 0), 3),
                                                            np.expand_dims(np.expand_dims(t1, 0), 3),
                                                            np.expand_dims(np.expand_dims(t2, 0), 3)])


                        pd = (pd - np.mean(pd)) / np.std(pd)
                        t1 = (t1 - np.mean(t1)) / np.std(t1)
                        t2 = (t2 - np.mean(t2)) / np.std(t2)

                        sfat = comps[0][0, :, :, 0]
                        sfat = np.interp(sfat, (sfat.min(), sfat.max()), (0, 1))
                        sfat = crop_image(sfat, mask, 0)
                        sfat = (sfat - np.mean(sfat)) / np.std(sfat)

                        swater = comps[1][0, :, :, 0]
                        swater = np.interp(swater, (swater.min(), swater.max()), (0, 1))
                        swater = crop_image(swater, mask, 0)
                        swater = (swater - np.mean(swater)) / np.std(swater)

                        np.savez(save_path + str.join("_", (patient, str(i))),
                                mr=np.array(mr, dtype=np.float32),
                                oop=np.array(oop, dtype=np.float32),
                                soop=np.array(soop, dtype=np.float32),
                                fat=np.array(fat, dtype=np.float32),
                                water=np.array(water, dtype=np.float32),
                                sfat=np.array(sfat, dtype=np.float32),
                                swater=np.array(swater, dtype=np.float32),
                                pd=np.array(pd, dtype=np.float32),
                                t1=np.array(t1, dtype=np.float32),
                                t2=np.array(t2, dtype=np.float32),
                                ct=np.array(ct, dtype=np.float32))
    patient_idx += 1
only_training = True