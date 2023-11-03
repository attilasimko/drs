import sys

sys.path.append("../")
import os
import random
from os.path import dirname, join

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from utils import transform_image_to_kspace, transform_kspace_to_image

random.seed(2019)
np.random.seed(2019)
import shutil
import sys

sys.path.append("/home/attilasimko/Documents/artefacts")
from corrupt import corrupt_image


def zero_pad(img):

    kspace = transform_image_to_kspace(img)
    smaller = False

    if ((np.shape(kspace)[0] >= 320) & (np.shape(kspace)[1] >= 320)):
        y,x = kspace.shape
        startx = x//2 - 320//2
        starty = y//2 - 320//2    
        kspace = kspace[starty:starty+320, startx:startx+320]
    elif ((np.shape(kspace)[0] <= 320) & (np.shape(kspace)[1] <= 320)):
        smaller = True
        shapex = ((int)((320 - np.shape(kspace)[0]) / 2), (int)((320 - np.shape(kspace)[0]) / 2))
        shapey = ((int)((320 - np.shape(kspace)[1]) / 2), (int)((320 - np.shape(kspace)[1]) / 2))
        kspace = np.pad(kspace, (shapex, shapey), 'constant', constant_values=(0, 0))
    
    img = transform_kspace_to_image(kspace)

    return img, smaller

data_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/fastMRI/brain_fastMRI_DICOM/fastMRI_brain_DICOM"

stackID = 0
wrongID = 0
ld = []
ln = []
lm = []
lb = []
ldvif = []
lnvif = []
lmvif = []
lbvif = []

base_path = '/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/DSets/DS0059/'
shutil.rmtree(base_path)
os.mkdir(base_path)
os.mkdir(base_path + "training")
os.mkdir(base_path + "validating")
os.mkdir(base_path + "testing")

patients = os.listdir(data_path)

np.random.shuffle(patients)
print(patients)

for patient in patients:
    is_pelvis = "Pelvis" in patient
    mult = 1
    STACK = []
    if ((patients.index(patient) / len(patients)) < 0.1):
        save_path =  base_path +'validating/'
    elif ((patients.index(patient) / len(patients)) < 0.3):
        save_path = base_path + 'testing/'
    else:
        mult = 1
        save_path = base_path + "training/"

    if (is_pelvis):
        for scan_file in os.listdir(os.path.join(data_path, patient, "MR")):
            data = pydicom.dcmread(os.path.join(data_path, patient, "MR", scan_file))
            STACK.append(data)
    else:
        for scan_file in os.listdir(os.path.join(data_path, patient)):
            data = pydicom.dcmread(os.path.join(data_path, patient, scan_file))
            STACK.append(data)

    try:
        STACK = sorted(STACK, key=lambda s: float(s.ImagePositionPatient[2]))
    except:
        print("No slice location")

    if (is_pelvis):
        STACK = STACK[0:50]

    for i in range(len(STACK)):
        if (not(os.path.exists(save_path + str.join("_", (patient, str(i), str(0))) + ".npz"))):
            image = STACK[i].pixel_array
            if (np.min(image) == np.max(image)):
                continue

            is_smaller = False
            if (np.shape(image) != (320, 320)):
                image, is_smaller = zero_pad(image)

            if (np.shape(image) == (320, 320)):
                for aug_i in range(mult):
                    if aug_i != 0:
                        image = np.roll(image, random.randint(-10, 10), 0)
                        image = np.roll(image, random.randint(-10, 10), 1)

                    image = (image - np.mean(image)) / np.std(image)

                    downsample = corrupt_image(np.copy(image), "downsample")
                    noise = corrupt_image(np.copy(image), "noise")
                    if (is_pelvis):
                        motion = corrupt_image(np.copy(image), "motion")
                    else:
                        motion = corrupt_image(np.copy(image), "motion_rigid")

                    bias = corrupt_image(np.copy(image), "bias")

                    if (not(is_smaller)):
                        mixed = corrupt_image(corrupt_image(corrupt_image(corrupt_image(np.copy(image), "downsample"), "noise"), "motion"), "bias")
                    else:
                        mixed = corrupt_image(corrupt_image(corrupt_image(np.copy(image), "noise"), "motion_rigid"), "bias")

                    ld.append(np.mean(np.square(image - downsample)))
                    ln.append(np.mean(np.square(image - noise)))
                    lm.append(np.mean(np.square(image - motion)))
                    lb.append(np.mean(np.square(image - bias)))

                    print("")
                    print(100 * (patients.index(patient) / len(patients)))
                    print(str(np.round(np.mean(ld), 4)) + "\pm" + str(np.round(np.std(ld), 4)))
                    print(str(np.round(np.mean(ln), 4)) + "\pm" + str(np.round(np.std(ln), 4)))
                    print(str(np.round(np.mean(lm), 4)) + "\pm" + str(np.round(np.std(lm), 4)))
                    print(str(np.round(np.mean(lb), 4)) + "\pm" + str(np.round(np.std(lb), 4)))

                    if (not(is_smaller)):
                        ind = random.choice([0, 2, 3])
                    else:
                        ind = random.choice([0, 1, 2, 3])

                    np.savez(save_path + str.join("_", (patient, str(i), str(aug_i))),
                            index = np.array(ind, dtype=np.int32),
                            clean=np.array(image, dtype=np.float32),
                            downsample=np.array(downsample, dtype=np.float32),
                            noise=np.array(noise, dtype=np.float32),
                            motion=np.array(motion, dtype=np.float32),
                            bias=np.array(bias, dtype=np.float32),
                            mixed=np.array(mixed, dtype=np.float32),
                            corrupt=np.array([bias, downsample, motion, noise][ind], dtype=np.float32))

            else:
                print(np.shape(image))