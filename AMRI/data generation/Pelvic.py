import sys
sys.path.append("../")
import scipy
import numpy as np
import os
from os.path import dirname, join
import random
import h5py
import tensorflow
import matplotlib.pyplot as plt
from matplotlib import cm
import nibabel as nib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import tarfile
import numpy
from matplotlib import pyplot, cm
import pydicom
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir
from scipy.stats import norm
import matplotlib
import matplotlib.mlab as mlab
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
import sys
sys.path.append("/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e")
from cv2 import resize
import shutil

sys.path.append("/home/attilasimko/Documents/artefacts")
from corrupt import corrupt_image

data_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/sCT_cured/" # Path to the folder containing MRI images.

base_path = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0057/'
shutil.rmtree(base_path)    
os.mkdir(base_path)
os.mkdir(base_path + "training")
os.mkdir(base_path + "validating")
os.mkdir(base_path + "testing")

stackID = 0
wrongID = 0
ld = []
ln = []
lm = []
lmx = []

patients = os.listdir(os.path.join(data_path))
for patient in os.listdir(os.path.join(data_path)):
    if (os.path.isdir(os.path.join(data_path, patient))):
        mult = 1
        unpaired = False
        STACK = []

        if ((patients.index(patient) / len(patients)) < 0.1):
            save_path =  base_path +'validating/'
        elif ((patients.index(patient) / len(patients)) < 0.3):
            save_path = base_path + 'testing/'
        else:
            mult = 1
            save_path = base_path + "training/"

        if (os.path.isdir(os.path.join(data_path, patient))):
            for contrast in os.listdir(os.path.join(data_path, patient)):
                if ((contrast == "MR") | (contrast == "MRI")):
                    for scan_file in os.listdir(os.path.join(data_path, patient, contrast)):
                        if scan_file.__contains__(".dcm"):
                            data = pydicom.dcmread(os.path.join(data_path, patient, contrast, scan_file))
                            STACK.append(data)

                    try:
                        STACK = sorted(STACK, key=lambda s: float(s.ImagePositionPatient[2]))
                    except:
                        print("No slice location")

                    for i in range(len(STACK)):
                        image = STACK[i].pixel_array
                        if (np.shape(image) != (320, 320)):
                            image = cv2.resize(image, (320, 320))

                        if (np.shape(image) == (320, 320)):
                            for aug_i in range(mult):
                                if aug_i != 0:
                                    image = np.roll(image, random.randint(-10, 10), 0)
                                    image = np.roll(image, random.randint(-10, 10), 1)

                                image = (image - np.mean(image)) / np.std(image)

                                downsample = corrupt_image(np.copy(image), "downsample")
                                noise = corrupt_image(np.copy(image), "noise")
                                motion = corrupt_image(np.copy(image), "motion")
                                bias = corrupt_image(np.copy(image), "bias")
                                mixed = corrupt_image(corrupt_image(corrupt_image(np.copy(image), "downsample"), "noise"), "motion")

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