from __future__ import print_function
from comet_ml import Experiment
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Code suppressing TF warnings and messages.
import numpy as np
import time
from tensorflow.keras import optimizers
from tensorflow.keras.backend import clear_session
import random
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
import argparse
import gc
import matplotlib.pyplot as plt
import tensorflow
from numpy.random import seed
from tensorflow import function, TensorSpec
from tensorflow import io
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import sys
from tensorflow.keras.utils import OrderedEnqueuer
import MLTK
from MLTK.synthetic_ct.models import build_discriminator, build_srresnet, build_unet
from MLTK.synthetic_ct.utils import get_patients, weighted_loss, KerasDropoutPrediction
from MLTK.data import DataGenerator
from sewar.full_ref import vifp
pid = os.getpid()
random.seed(2021)
seed(2021)
print(pid)

def compare_mae(pred, gt):
    return np.abs(pred - gt)

def compare_mse(pred, gt):
    return np.square(pred - gt)

def compare_re(pred, gt):
    return np.abs((pred - gt) / gt)

parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--lr", default=0.0001, help="Learning rate for the model.")
parser.add_argument("--case", default="III", help="Training scenarios. I for training on MR, II for training on sQM.")
parser.add_argument("--dropout", default=0.2, help="Dropout rate.")
parser.add_argument("--batch_size", default=2, help="Batch size for training.")
parser.add_argument("--gpu", default=None, help="Set the index of the GPU (if there is multiple).")

args = parser.parse_args()

batchnorm = False
num_filters = 64
num_res_block = 12

gpu = args.gpu
lr = float(args.lr)
case = args.case

dropout_rate = float(args.dropout)
batch_size = int(args.batch_size)

data_path = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0060/'
out_path = '/home/attilasimko/Documents/out/'

physical_devices = tensorflow.config.list_physical_devices('GPU')

if (case == "I"):
    num_inputs = 1
    num_outputs = 1
    gen_train = DataGenerator(data_path + "training",
                        inputs=[['mr', False, 'float32']],
                        outputs=[['ct', False, 'float32']],
                        batch_size=batch_size,
                        shuffle=True)

    gen_val = DataGenerator(data_path + 'validating',
                        inputs=[['mr', False, 'float32']],
                        outputs=[['ct', False, 'float32']],
                        batch_size=1,
                        shuffle=False)

    gen_test = DataGenerator(data_path + 'testing',
                        inputs=[['mr', False, 'float32']],
                        outputs=[['ct', False, 'float32']],
                        batch_size=1,
                        shuffle=False)
    
    generator = build_srresnet(num_filters=num_filters, batchnorm=batchnorm, case=case, dropout_rate=dropout_rate, num_inputs=num_inputs, num_outputs=num_outputs)
    generator.compile(optimizer=optimizers.Adam(lr), loss=["mse"], run_eagerly=True)
elif (case == "II"):
    num_inputs = 3
    num_outputs = 1
    gen_train = DataGenerator(data_path + "training",
                        inputs=[['pd', False, 'float32'],
                                ['t1', False, 'float32'],
                                ['t2', False, 'float32']],
                        outputs=[['ct', False, 'float32']],
                        batch_size=batch_size,
                        shuffle=True)

    gen_val = DataGenerator(data_path + 'validating',
                        inputs=[['pd', False, 'float32'],
                                ['t1', False, 'float32'],
                                ['t2', False, 'float32']],
                        outputs=[['ct', False, 'float32']],
                        batch_size=1,
                        shuffle=False)

    gen_test = DataGenerator(data_path + 'testing',
                        inputs=[['pd', False, 'float32'],
                                ['t1', False, 'float32'],
                                ['t2', False, 'float32']],
                        outputs=[['ct', False, 'float32']],
                        batch_size=1,
                        shuffle=False)

    generator = build_srresnet(num_filters=num_filters, batchnorm=batchnorm, case=case, dropout_rate=dropout_rate, num_inputs=num_inputs, num_outputs=num_outputs)
    generator.compile(optimizer=optimizers.Adam(lr), loss=["mse"], run_eagerly=True)

num_workers = 2

generator_kdp = KerasDropoutPrediction(generator)
patience_thr = 50
epoch_thr = 50

patience = 0
best_loss = np.inf
n_epochs = 50

validation_patients = get_patients(gen_val)

for epoch in range(n_epochs):
    tensorflow.keras.backend.clear_session()

    tr_seq = OrderedEnqueuer(gen_train, use_multiprocessing=False)
    val_seq = OrderedEnqueuer(gen_val, use_multiprocessing=False)

    ct_unc_list = []
    uncertainty_corr_mae_list = []
    uncertainty_corr_mse_list = []
    uncertainty_corr_re_list = []

    gan_loss_list = []
    ct_loss_mae_list = []
    ct_loss_mse_list = []
    ct_loss_re_list = []


    ct_loss_vif_list = []
    uncertainty_corr = []

    tr_seq.start(workers=num_workers, max_queue_size=10)
    data_seq = tr_seq.get()
    len_train = int(len(gen_train))
    for idx in range(len_train):
        x_mri, x_ct = next(data_seq)
        gan_loss = generator.train_on_batch(x_mri, x_ct)
        gan_loss_list.append(gan_loss)

    gen_train.on_epoch_end()
    tr_seq.stop()

    val_seq.start(workers=num_workers, max_queue_size=10)
    data_seq = val_seq.get()
    len_val = len(gen_val)
    for idx in range(len_val):
        x_mri, x_ct = next(data_seq)

        pred = generator.predict_on_batch(x_mri)
        unc = generator_kdp.predict(x_mri).numpy()

        
        mae = 0
        mse = 0
        re = 0
        for i in range(num_outputs):
            mae += compare_mae(pred[i], x_ct[i])
            mse += compare_mse(pred[i], x_ct[i])
            re += compare_re(pred[i], x_ct[i])
        

        ct_loss_mae_list.append(np.average(mae, axis=(1,2,3)))
        ct_loss_mse_list.append(np.average(mse, axis=(1,2,3)))
        ct_loss_re_list.append(np.average(re, axis=(1,2,3)))

        ct_unc_list.append(np.average(unc))

        uncertainty_corr_mae_list.append(np.corrcoef(np.ndarray.flatten(unc), np.ndarray.flatten(mae))[0, 1])
        uncertainty_corr_mse_list.append(np.corrcoef(np.ndarray.flatten(unc), np.ndarray.flatten(mse))[0, 1])
        uncertainty_corr_re_list.append(np.corrcoef(np.ndarray.flatten(unc), np.ndarray.flatten(re))[0, 1])
        
    gen_val.on_epoch_end()
    val_seq.stop()

    print(f"Epoch {epoch}\ttraining: {np.mean(gan_loss_list)}\tvalidating: {np.mean(ct_loss_mae_list)}")

    if (best_loss > (np.mean(ct_loss_mae_list))):
        patience = 0
        best_loss = (np.mean(ct_loss_mae_list))
        for patient in validation_patients:
            data = np.zeros((3, 200, 512, 512, 1))
            ct_data = np.zeros((200, 512, 512, 1))
            pred = np.zeros((200, 512, 512, 1))
            pred_unc = np.zeros((200, 512, 512, 1))
            plot_z = 0
            gen_val.on_epoch_end()
            for idx in range(len(gen_val)):
                if patient == gen_val.file_list[idx].split('/')[-1].split(' ')[0]:
                    x_mri, x_ct = gen_val[idx]
                    slc = int(gen_val.file_list[idx].split('_')[-1].split('.')[0])
                    
                    data_slice = x_mri

                    unc_slice = generator_kdp.predict(data_slice)
                    pred_slice = generator.predict_on_batch(x_mri)
                    for i in range(3):
                        if (i < num_inputs):
                            data[i, slc, :, :, 0] = data_slice[i][0, :, :, 0]
                    ct_data[slc, :, :, 0] = x_ct[0][0, :, :, 0]
                    pred_unc[slc, :, :, 0] = unc_slice[0, :, :, 0]
                    
                    if (num_outputs > 1):
                        pred[slc, :, :, 0] = pred_slice[0][0, :, :, 0]
                    else:
                        pred[slc, :, :, 0] = pred_slice[0, :, :, 0]
                    plot_z += 1

            ax_slice = 250
            sag_slice = 150
            cor_slice = 15


            data = data[:, 0:plot_z, :, :, :]
            ct_data = ct_data[0:plot_z, :, :, :]
            pred = pred[0:plot_z, :, :, :]
            pred_unc = pred_unc[0:plot_z, :, :, :]

            diff_max = 1.0
            unc_max = 0.2

            plt.figure(figsize=(56, 24))
            for i in range(3):
                plt.subplot(3, 7, 1 + i)
                plt.imshow(data[i, :, ax_slice, :, 0], cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(3, 7, 8 + i)
                plt.imshow(data[i, :, :, sag_slice, 0], cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(3, 7, 15 + i)
                plt.imshow(data[i, cor_slice, :, :, 0], cmap='gray')
                plt.xticks([])
                plt.yticks([])


            plt.subplot(3, 7, 4)
            plt.imshow(pred[:, ax_slice, :, 0], vmin=-1, vmax=1, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 7, 5)
            plt.imshow(ct_data[:, ax_slice, :, 0], vmin=-1, vmax=1, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 7, 6)
            plt.imshow(np.abs(ct_data[:, ax_slice, :, 0] - pred[:, ax_slice, :, 0]), vmin=0, vmax=diff_max, cmap='Reds')
            plt.xticks([])
            plt.yticks([])
            #plt.colorbar()
            plt.subplot(3, 7, 7)
            plt.imshow(pred_unc[:, ax_slice, :, 0], vmin=0, vmax=unc_max, cmap='Reds')
            plt.xticks([])
            plt.yticks([])
            #plt.colorbar()


            plt.subplot(3, 7, 11)
            plt.imshow(pred[:, :, sag_slice, 0], vmin=-1, vmax=1, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 7, 12)
            plt.imshow(ct_data[:, :, sag_slice, 0], vmin=-1, vmax=1, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 7, 13)
            plt.imshow(np.abs(ct_data[:, :, sag_slice, 0] - pred[:, :, sag_slice, 0]), vmin=0, vmax=diff_max, cmap='Reds')
            plt.xticks([])
            plt.yticks([])
            #plt.colorbar()
            plt.subplot(3, 7, 14)
            plt.imshow(pred_unc[:, :, sag_slice, 0], vmin=0, vmax=unc_max, cmap='Reds')
            plt.xticks([])
            plt.yticks([])
            #plt.colorbar()


            plt.subplot(3, 7, 18)
            plt.imshow(pred[cor_slice, :, :, 0], vmin=-1, vmax=1, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 7, 19)
            plt.imshow(ct_data[cor_slice, :, :, 0], vmin=-1, vmax=1, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 7, 20)
            plt.imshow(np.abs(ct_data[cor_slice, :, :, 0] - pred[cor_slice, :, :, 0]), vmin=0, vmax=diff_max, cmap='Reds')
            plt.xticks([])
            plt.yticks([])
            #plt.colorbar()
            plt.subplot(3, 7, 21)
            plt.imshow(pred_unc[cor_slice, :, :, 0], vmin=0, vmax=unc_max, cmap='Reds')
            plt.xticks([])
            plt.yticks([])
            #plt.colorbar()
            plt.savefig(out_path + "pics/" + str(patient) + ".svg")
            plt.close('all')

        generator.save_weights(str(out_path) + str(pid) + '.h5')
    else:
        patience += 1
        if ((patience > patience_thr) | (epoch > epoch_thr)):
            break
    
    gc.collect()
    tensorflow.keras.backend.clear_session()


generator.load_weights(str(out_path) + str(pid) + '.h5')

def test_gen(gen):
    a_list = []
    st_list = []
    b_list = []
    vif_list = []
    for idx in range(len(gen)):
        x_mri, x_ct = gen[idx]

        pred = generator.predict_on_batch(x_mri)
        if (num_outputs == 1):
            pred = [pred]
        
        for i in range(num_outputs):
            vif_list.append(vifp(pred[i][0, :, :, 0], x_ct[i][0, :, :, 0]))
            a_list.append(np.average(np.abs(pred[i][0, :, :, 0] - x_ct[i][:, :, :, 0]), weights=((x_ct[i][:, :, :, 0] > -1) * (x_ct[i][:, :, :, 0] <= -0.1))))
            st_list.append(np.average(np.abs(pred[i][0, :, :, 0] - x_ct[i][:, :, :, 0]), weights=((x_ct[i][:, :, :, 0] > -0.1) * (x_ct[i][:, :, :, 0] <= 0.1))))
            b_list.append(np.average(np.abs(pred[i][0, :, :, 0] - x_ct[i][:, :, :, 0]), weights=((x_ct[i][:, :, :, 0] > 0.1) * (x_ct[i][:, :, :, 0] <= 1))))

        vif_res = (str(np.round(np.nanmean(vif_list), 5)) + " +- " + str(np.round(np.nanstd(vif_list) / np.sqrt(len(vif_list)), 10)))
        a_res = (str(np.round(np.nanmean(a_list), 5)) + " +- " + str(np.round(np.nanstd(a_list) / np.sqrt(len(a_list)), 10)))
        st_res = (str(np.round(np.nanmean(st_list), 5)) + " +- " + str(np.round(np.nanstd(st_list) / np.sqrt(len(st_list)), 10)))
        b_res = (str(np.round(np.nanmean(b_list), 5)) + " +- " + str(np.round(np.nanstd(b_list) / np.sqrt(len(b_list)), 10)))
    return (a_res, st_res, b_res, vif_res)

(a_res, st_res, b_res, vif_res) = test_gen(gen_test)
print("\nBaseline:")
print(a_res)
print(st_res)
print(b_res)
print(vif_res)