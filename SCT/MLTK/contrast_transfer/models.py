# -*- coding: utf-8 -*-
"""
This module contains ready-to-use Keras-like and Keras-compatible models.

Created on Mon Oct  9 13:48:25 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc
import json
import six
import warnings

import h5py
import numpy as np

import tensorflow
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.python.ops.gen_nn_ops import Softmax
from MLTK.data import DataGenerator
from MLTK.utils import Timer, new_divide, MyCustomCallback, WeightNormalization
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os


__all__ = ["BaseModel",
           "ConvolutionalNetwork", "FullyConvolutionalNetwork", "UNet", "GAN"]



class MTNN(object):   
    def __init__(self, comet, lr_gen, lr_gan, lr_dis, update_it=1, gamma = 0, beta=0, std=0.0001, optimizer='Adam', clipnorm=True, loss='categorical_crossentropy', df=32, paired=False, g_path=None, d_path=None, save_dir='out/', batch_size=16, gen_dir=None):
        import tensorflow
        from MLTK.data import DataGenerator
        from MLTK.utils import RandomWeightedAverage, wasserstein_loss, smooth, gradient_penalty
        import tensorflow
        from tensorflow.keras.models import load_model
        from tensorflow.keras.optimizers import Adam, Nadam
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        from tensorflow.keras.utils import OrderedEnqueuer
        self.TFReset()
        self.learning_rate_gen = lr_gen
        self.learning_rate_gan = lr_gan
        self.learning_rate_dis = lr_dis
        self.save_dir = save_dir
        self.comet = comet
        self.paired = paired
        self.batch_size = batch_size
        self.gen_dir = gen_dir
        self.update_it = update_it
        self.beta = beta
        self.optimizer = optimizer
        self.clipnorm = clipnorm
        self.loss = loss
        self.std = std
        self.gamma = gamma
        self.df = df
        self.alpha = 0.001
        self.weight_te = 1
        self.weight_tr = 1

        rows, cols, channels = 256, 256, 1

        if gen_dir != None:
            self.tr_gen_p = DataGenerator(gen_dir+'training_p',
                                        inputs=[['im1', False, 'float32'],
                                                ['ter1', False, 'int8']],
                                        outputs=[['im2', False, 'float32'],
                                                ['ter2', False, 'int8']],
                                        batch_size=batch_size,
                                        shuffle=True)

            self.tr_gen = DataGenerator(gen_dir+'training_fov',
                                        inputs=[['im1', False, 'float32'],
                                                ['ter1', False, 'int']],
                                        outputs=[],
                                        batch_size=batch_size,
                                        shuffle=True)

            self.val_gen = DataGenerator(gen_dir+'validating_ter_new',
                                        inputs=[['im1', False, 'float32'],
                                                ['im2', False, 'float32'],
                                                ['im3', False, 'float32'],
                                                ['im4', False, 'float32'],
                                                ['im5', False, 'float32']],
                                        outputs=[['PD', False, 'float32'],
                                                ['T1', False, 'float32'],
                                                ['T2', False, 'float32'],
                                                ['mask', False, 'bool']],
                                        batch_size=25,
                                        shuffle=False)
            self.test_gen = DataGenerator(gen_dir+'testing_lsq',
                                        inputs=[['im1', False, 'float32'],
                                                ['im2', False, 'float32'],
                                                ['im3', False, 'float32'],
                                                ['im4', False, 'float32'],
                                                ['im5', False, 'float32']],
                                        outputs=[['T1', False, 'float32'],
                                                ['T2', False, 'float32'],
                                                ['mask', False, 'bool']],
                                        batch_size=batch_size,
                                        shuffle=False)


            self.tr_seq = OrderedEnqueuer(self.tr_gen, use_multiprocessing=False)
            # self.tr_pair_seq = OrderedEnqueuer(self.tr_gen_p, use_multiprocessing=False)
            self.val_seq = OrderedEnqueuer(self.val_gen, use_multiprocessing=False)
        
        self.opt_gen = Nadam(lr=self.learning_rate_gen)

        if self.optimizer == 'Adam':
            self.opt_gan = Adam(lr=self.learning_rate_gan, beta_1=self.beta, beta_2=0.999, clipnorm=self.clipnorm) #, beta_1=0.9, beta_2=0.999) # Try 0.9, 0.999
            self.opt_dis = Adam(lr=self.learning_rate_dis, beta_1=self.beta, beta_2=0.999, clipnorm=self.clipnorm) # with default here
        elif self.optimizer == 'Nadam':
            self.opt_gan = Nadam(lr=self.learning_rate_gan, beta_1=self.beta, beta_2=0.999, clipnorm=self.clipnorm) #, beta_1=0.9, beta_2=0.999) # Try 0.9, 0.999
            self.opt_dis = Nadam(lr=self.learning_rate_dis, beta_1=self.beta, beta_2=0.999, clipnorm=self.clipnorm) # with default here
        
        self.discriminator = self.build_discriminator(self.df, rows, cols, channels)
        # Build and compile the discriminator
        if d_path != None:
            self.discriminator.load_weights(d_path)

        # Build and compile the generator
        self.generator = self.build_generator()
        if g_path != None:
            self.generator.load_weights(g_path)
            
        self.discriminator_fake = Model(inputs=[self.discriminator.inputs[0]], 
                            outputs=[self.discriminator.outputs[2]])
        self.discriminator_ter = Model(inputs=[self.discriminator.inputs[0]], 
                            outputs=[self.discriminator.outputs[0], self.discriminator.outputs[0]])
        # Component generator   
        self.generator.compile(optimizer=self.opt_gen, loss=['mean_squared_error'], metrics=['mean_squared_error'])
        self.discriminator.compile(optimizer=self.opt_dis, loss=[self.loss, self.loss, 'mean_squared_error'], metrics=['accuracy'], loss_weights=[self.weight_te, self.weight_tr, 1]) # binary_crossentropy
        self.discriminator_fake.compile(optimizer=self.opt_dis, loss=['mean_squared_error'], metrics=['accuracy'])
        self.discriminator_ter.compile(optimizer=self.opt_dis, loss=[self.loss, self.loss], metrics=['accuracy'], loss_weights=[self.weight_te, self.weight_tr])
        # self.discriminator_fake = Model(inputs=[self.discriminator.inputs[0]],
        #                                 outputs=[self.discriminator.outputs[0]])
        # self.discriminator_fake.compile(optimizer=self.opt_dis, loss=[self.loss], metrics=[tensorflow.keras.metrics.BinaryAccuracy()]) # binary_crossentropy

        # self.discriminator_ter = Model(inputs=[self.discriminator.inputs[0]],
        #                                 outputs=[self.discriminator.outputs[1], self.discriminator.outputs[2]])
        # self.discriminator_ter.compile(optimizer=self.opt_dis, loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[self.weight_te, self.weight_tr])

        
        # GAN
        img_in = Input(shape=(rows, cols, channels), batch_size=self.batch_size)
        ter_in = Input(shape=(2), batch_size=self.batch_size)
        ter_out = Input(shape=(2), batch_size=self.batch_size)
        for layer in self.discriminator.layers:
            layer.trainable = False
        pred_mt = self.generator([img_in, ter_out])
        pred_unit = self.generator([pred_mt, ter_in])
        pred_d = self.discriminator([pred_mt])
        self.GAN = Model(inputs=[img_in, ter_out], outputs=[pred_d[0], pred_d[1], pred_d[2]])
        self.GAN.compile(optimizer=self.opt_gan, loss=['mean_squared_error', self.loss, self.loss], metrics=['accuracy'], loss_weights=[self.weight_te, self.weight_tr, 1]) # binary_crossentropy

    def TFReset(self):
        # import gc
        tensorflow.keras.backend.clear_session()
        np.random.seed(113)
        tensorflow.random.set_seed(113)
        # gc.collect()

    def train(self, train_steps=10, show_every=0):
        # GAN initialization
        import random
        from tensorflow.keras.utils import to_categorical, OrderedEnqueuer
        from MLTK.utils import smooth
        from MLTK.bias_field_correction.utils import mse_mask
        from sklearn.metrics import confusion_matrix
        from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity, normalized_root_mse
        from numpy.random import uniform
        import gc



        self.ae = True

        base_ter = np.array([[0.075, 4.5],  [0.12, 4.5] , [0.008, 0.4], [0.008, 0.75], [0.008, 4.5], [0, 0]], dtype=np.float32)
        base_ter = np.array([base_ter for p in range(self.batch_size)])
        weight_keep = np.ones((self.batch_size))
        weight_skip = np.zeros((self.batch_size))

        optimizer_loss = []

        dummy_gp = np.zeros(shape=(self.batch_size, 6))
        
        go_gan = False
        pretrain = True
        pretrain_fake = False
        pretrain_ter = True
        min_loss = np.inf
        num_workers = 1

        for epoch in range(train_steps):
            if self.comet:
                self.comet.set_step(epoch)
            # Log training
            loss_d = []
            loss_d_fake = []
            loss_d_te = []
            loss_d_tr = []
            acc_d = []
            loss_gan = []
            loss_gan_te = []
            loss_gan_tr = []
            loss_gan_cycle = []
            acc_gan = []
            loss_g = []

            # Log validating
            val_loss_d = []
            val_loss_d_te = []
            val_loss_d_tr = []
            val_acc_d = []
            val_loss_d_fake = []
            val_loss_d_te_fake = []
            val_loss_d_tr_fake = []
            val_acc_d_fake = []
            val_loss_gan = []
            val_loss_gan_te = []
            val_loss_gan_tr = []
            val_loss_gan_cycle = []
            val_acc_gan = []

            val_mse_mt = []
            val_nrmse_mt = []
            val_psnr_mt = []
            val_ssim_mt = []
            val_mse_ae = []
            val_nrmse_ae = []
            val_psnr_ae = []
            val_ssim_ae = []

            val_loss_t1 = []
            val_loss_t2 = []



            # TRAINING
            timer = Timer()

            # Cycle            
            self.tr_seq.start(workers=num_workers, max_queue_size=5)
            data_seq = self.tr_seq.get()
            # self.tr_pair_seq.start(workers=num_workers, max_queue_size=5)
            # data_seq_p = self.tr_pair_seq.get()
            self.TFReset()
            real_class = np.ones((self.tr_gen.batch_size, 1))
            fake_class = np.zeros((self.tr_gen.batch_size, 1))

            for idx in range(int(5000)):
                
                x, y = next(data_seq)
                in_img, in_idx = x
                in_te, in_tr = self.make_map(in_idx[:, 0])
                in_te_g = np.interp(in_te, (0, 1), (0.008, 0.12))
                in_tr_g = np.interp(in_tr, (0, 1), (0.4, 4.5))

                # out_te, out_tr = self.make_fake_map(in_idx[:, 0])
                # out_te_g = np.interp(out_te, (0, 1), (0.008, 0.12))
                # out_tr_g = np.interp(out_tr, (0, 1), (0.4, 4.5))
                # pred = self.generator.predict_on_batch([in_img, np.concatenate((out_te_g, out_tr_g), axis=1)])
                # loss = self.discriminator_ter.train_on_batch(in_img, [in_te, in_tr])
                # loss_d_te.append(self.weight_te * loss[1])
                # loss_d_tr.append(self.weight_tr * loss[2])
                # acc_d.append(loss[3])
                # loss = self.discriminator_fake.train_on_batch(np.concatenate((in_img, pred), axis=0), np.concatenate((real_class, fake_class), axis=0))
                # acc_d.append(loss[0])
                
                if (pretrain):
                    x, y = self.tr_gen_p[idx]
                    in_p, in_idx_p = x
                    out_p, out_idx_p = y
                    out_te_p, out_tr_p = self.make_map(out_idx_p[:, 0])
                    out_te_g_p = np.interp(out_te_p, (0, 1), (0.008, 0.12))
                    out_tr_g_p = np.interp(out_tr_p, (0, 1), (0.4, 4.5))
                    loss = self.generator.train_on_batch([in_p, np.concatenate((out_te_g_p, out_tr_g_p), axis=1)], out_p)
                    loss_g.append(loss[0])
                
                if  (go_gan) & (idx % self.update_it == 0):
                    out_te, out_tr = self.make_fake_map(in_idx[:, 0])
                    out_te_g = np.interp(out_te, (0, 1), (0.008, 0.12))
                    out_tr_g = np.interp(out_tr, (0, 1), (0.4, 4.5))
                    loss = self.GAN.train_on_batch([in_img, np.concatenate((out_te_g, out_tr_g), axis=1)], [out_te, out_tr, real_class])
                    loss_gan_te.append(self.weight_te * loss[1])
                    loss_gan_tr.append(self.weight_tr * loss[2])
                    acc_gan.append(loss[3])
                    # loss_gan_cycle.append(self.alpha * loss[3])
            
            self.tr_seq.stop()
            self.tr_gen.on_epoch_end()
            # self.tr_pair_seq.stop()
            self.tr_gen_p.on_epoch_end()

            with np.errstate(invalid='ignore'):
                if self.comet:
                    self.comet.log_metrics({'D_loss': round(np.mean(loss_d), 5),
                                            'D_te': round(np.mean(loss_d_te), 5),
                                            'D_tr': round(np.mean(loss_d_tr), 5),
                                            'D_fake': round(np.mean(loss_d_fake), 5),
                                            'D_acc': round(np.mean(acc_d), 5),
                                            'GAN_loss': round(np.mean(loss_gan), 5),
                                            'GAN_te': round(np.mean(loss_gan_te), 5),
                                            'GAN_tr': round(np.mean(loss_gan_tr), 5),
                                            'GAN_cycle': round(np.mean(loss_gan_cycle), 5),
                                            'GAN_acc': round(np.mean(acc_gan), 5),
                                            'G_loss': round(np.mean(loss_g), 5)})


            # VALIDATION
            self.val_seq.start(workers=num_workers, max_queue_size=5)
            data_seq = self.val_seq.get()
            test_generator = Model(inputs=[self.generator.inputs[0]], 
                            outputs=[self.generator.get_layer('pd').output,
                            self.generator.get_layer('t1').output,
                            self.generator.get_layer('t2').output])
        
            # Compile Models
            test_generator.compile(loss='mean_squared_error')   

            real_class = np.ones((self.val_gen.batch_size, 1))
            fake_class = np.zeros((self.val_gen.batch_size, 1))
            for idx in range(len(self.val_gen)):
                self.TFReset()
                x, y = next(data_seq)
                # for i in range(5):
                #     x[i] = (x[i] - np.mean(x[i], axis=(1, 2, 3))[:, None, None, None]) / np.std(x[i], axis=(1, 2, 3))[:, None, None, None]

                for i in range(5):
                    i_map = np.array([i for p in range(self.val_gen.batch_size)])
                    in_te, in_tr = self.make_map(i_map)
                    in_te_g = np.interp(in_te, (0, 1), (0.008, 0.12))
                    in_tr_g = np.interp(in_tr, (0, 1), (0.4, 4.5))
                    decomp = test_generator.predict_on_batch(x[i])

                    for loss_idx in range(self.val_gen.batch_size):
                        temp_mask = (y[1][loss_idx, :, :, 0, 0] != np.min(y[1][loss_idx, :, :, 0, 0]))
                        val_loss_t1.append(mse_mask(decomp[1][loss_idx, :, :, 0], y[1][loss_idx, :, :, 0, 0], temp_mask))
                        val_loss_t2.append(mse_mask(decomp[2][loss_idx, :, :, 0], y[2][loss_idx, :, :, 0, 0], temp_mask))

                    loss = self.discriminator.test_on_batch(x[i], [in_te, in_tr, real_class])
                    val_acc_d.append(loss[-1])
                    val_loss_d_te.append(loss[1])
                    val_loss_d_tr.append(loss[2])

                    for j in range(5):
                        j_map = np.array([j for p in range(self.val_gen.batch_size)])
                        out_te, out_tr = self.make_map(j_map)
                        out_te_g = np.interp(out_te, (0, 1), (0.008, 0.12))
                        out_tr_g = np.interp(out_tr, (0, 1), (0.4, 4.5))

                        loss = self.GAN.test_on_batch([x[i], np.concatenate((out_te_g, out_tr_g), axis=1)], [out_te, out_tr, real_class])
                        val_loss_gan_te.append(loss[1])
                        val_loss_gan_tr.append(loss[2])
                        val_acc_gan.append(loss[-1])
                        # val_loss_gan_cycle.append(loss[3])

                        pred = np.array(self.generator.predict_on_batch([x[i], np.concatenate((out_te_g, out_tr_g), axis=1)]))
                        loss = self.discriminator.test_on_batch(pred, [out_te, out_tr, fake_class])
                        val_loss_d_te_fake.append(loss[1])
                        val_loss_d_tr_fake.append(loss[2])
                        val_acc_d_fake.append(loss[-1])

                        for loss_idx in range(self.val_gen.batch_size):
                            mask = y[3][loss_idx, :, :, 0]
                            if i == j:
                                val_mse_ae.append(mean_squared_error(pred[loss_idx, :, :, 0] * mask, x[j][loss_idx, :, :, 0] * mask))
                                val_nrmse_ae.append(normalized_root_mse(pred[loss_idx, :, :, 0] * mask, x[j][loss_idx, :, :, 0] * mask))
                                val_psnr_ae.append(peak_signal_noise_ratio(pred[loss_idx, :, :, 0] * mask, x[j][loss_idx, :, :, 0] * mask, data_range=np.ceil(np.max([pred[loss_idx, :, :, 0], x[j][loss_idx, :, :, 0]]) - np.min([pred[loss_idx, :, :, 0], x[j][loss_idx, :, :, 0]]))))
                                val_ssim_ae.append(structural_similarity(pred[loss_idx, :, :, 0] * mask, x[j][loss_idx, :, :, 0] * mask, data_range=np.ceil(np.max([pred[loss_idx, :, :, 0], x[j][loss_idx, :, :, 0]]) - np.min([pred[loss_idx, :, :, 0], x[j][loss_idx, :, :, 0]]))))
                            else:
                                val_mse_mt.append(mean_squared_error(pred[loss_idx, :, :, 0] * mask, x[j][loss_idx, :, :, 0] * mask))
                                val_nrmse_mt.append(normalized_root_mse(pred[loss_idx, :, :, 0] * mask, x[j][loss_idx, :, :, 0] * mask))
                                val_psnr_mt.append(peak_signal_noise_ratio(pred[loss_idx, :, :, 0] * mask, x[j][loss_idx, :, :, 0] * mask, data_range=np.ceil(np.max([pred[loss_idx, :, :, 0], x[j][loss_idx, :, :, 0]]) - np.min([pred[loss_idx, :, :, 0], x[j][loss_idx, :, :, 0]]))))
                                val_ssim_mt.append(structural_similarity(pred[loss_idx, :, :, 0] * mask, x[j][loss_idx, :, :, 0] * mask, data_range=np.ceil(np.max([pred[loss_idx, :, :, 0], x[j][loss_idx, :, :, 0]]) - np.min([pred[loss_idx, :, :, 0], x[j][loss_idx, :, :, 0]]))))




            self.val_seq.stop()
            self.val_gen.on_epoch_end()
            with np.errstate(invalid='ignore'):
                if self.comet:
                    self.comet.log_metrics({'vD_loss': round(np.mean(val_loss_d), 5),
                                            'vD_te': round(np.mean(val_loss_d_te), 5),
                                            'vD_tr': round(np.mean(val_loss_d_tr), 5),
                                            'vD_acc': round(np.mean(val_acc_d), 5),
                                            'vD_loss_fake': round(np.mean(val_loss_d_fake), 5),
                                            'vD_te_fake': round(np.mean(val_loss_d_te_fake), 5),
                                            'vD_tr_fake': round(np.mean(val_loss_d_tr_fake), 5),
                                            'vD_acc_fake': round(np.mean(val_acc_d_fake), 5),
                                            'vGAN_loss': round(np.mean(val_loss_gan), 5),
                                            'vGAN_te': round(np.mean(val_loss_gan_te), 5),
                                            'vGAN_tr': round(np.mean(val_loss_gan_tr), 5),
                                            'vGAN_cycle': round(np.mean(val_loss_gan_cycle), 5),
                                            'vGAN_acc': round(np.mean(val_acc_gan), 5),
                                            'vT1': round(np.mean(val_loss_t1), 5),
                                            'vT2': round(np.mean(val_loss_t2), 5),
                                            'vMT_mse': round(np.mean(val_mse_mt), 5),
                                            'vMT_nrmse': round(np.mean(val_nrmse_mt), 5),
                                            'vMT_psnr': round(np.mean(val_psnr_mt), 5),
                                            'vMT_ssim': round(np.mean(val_ssim_mt), 5),
                                            'vAE_mse': round(np.mean(val_mse_ae), 5),
                                            'vAE_nrmse': round(np.mean(val_nrmse_ae), 5),
                                            'vAE_psnr': round(np.mean(val_psnr_ae), 5),
                                            'vAE_ssim': round(np.mean(val_ssim_ae), 5)})

                if ((np.mean(val_mse_mt)) < min_loss):
                    min_loss = np.mean(val_mse_mt)
                    print("New best MSE loss: " + str(min_loss))
                    self.print_results(epoch)
                    self.generator.save_weights(self.save_dir + "best_g.h5")
                    self.discriminator.save_weights(self.save_dir + "best_d.h5")
                
            optimizer_loss.append(round(np.mean(val_mse_mt), 5))
            timer.elapsed_time()

            if (pretrain_ter & (not(go_gan)) & (np.mean(val_loss_d_te) <= 0.1) & (np.mean(val_loss_d_tr) <= 0.2) & (np.mean(val_acc_d) >= 0.9) & (np.mean(val_acc_d_fake) >= 0.9)):
                pretrain_ter = False

            if ((not(pretrain)) & (not(go_gan)) & (not(pretrain_fake)) & (not(pretrain_ter))): # 0.025 0.75
                go_gan = True
                print("Go GAN")
        return np.min(optimizer_loss) 

    def make_map(self, vector):
        out_te = np.ones((np.shape(vector)[0], 1))
        out_tr = np.ones((np.shape(vector)[0], 1))
        base_te = [0.075,  0.12, 0.008, 0.008, 0.008]
        base_tr = [4.5,  4.5,  0.4,  0.75,  4.5]
        for i in range(np.shape(vector)[0]):
            out_te[i, 0] = base_te[vector[i]]
            out_tr[i, 0] = base_tr[vector[i]]
        return np.interp(out_te, (0.008, 0.12), (0, 1)), np.interp(out_tr, (0.4, 4.5), (0, 1))

    def make_fake_map(self, vector):
        out_te = np.ones((np.shape(vector)[0], 1))
        out_tr = np.ones((np.shape(vector)[0], 1))
        base_te = [0.075,  0.12, 0.008, 0.008, 0.008]
        base_tr = [4.5,  4.5,  0.4,  0.75,  4.5]
        idx = np.random.randint(4)
        for i in range(np.shape(vector)[0]):
            out_te[i, 0] = np.delete(base_te, vector[i])[idx]
            out_tr[i, 0] = np.delete(base_tr, vector[i])[idx]
        
        out_te = np.interp(out_te, (0.008, 0.12), (0, 1))
        out_tr = np.interp(out_tr, (0.4, 4.5), (0, 1))
        # out_te = np.clip(out_te + np.random.normal(scale=0.01), 0, 1)
        # out_tr = np.clip(out_tr + np.random.normal(scale=0.01), 0, 1)
        return out_te, out_tr

    def to_ter(self, te, tr):
        batch_size = len(te)
        base = [0.075 +  4.5, 0.12 +  4.5, 0.008 +  0.4, 0.008 +  0.75, 0.008 +  4.5]
        base = np.array([base for p in range(batch_size)]) 
        ter = te + tr
        ter = np.repeat(ter, 5, axis=1)
        ter = np.isclose(ter, base)
        if np.sum(ter) != batch_size:
            return None
        return ter

    def to_te_tr(self, ter):
        ter = np.isclose(ter, 1)
        batch_size = len(ter)
        base_te = [0.075,  0.12, 0.008, 0.008, 0.008]
        base_tr = [4.5,  4.5,  0.4,  0.75,  4.5]
        base_te = np.array([base_te for p in range(batch_size)]) 
        base_tr = np.array([base_tr for p in range(batch_size)]) 
        base_te = np.array(base_te[ter], dtype=np.float32)
        base_tr = np.array(base_tr[ter], dtype=np.float32)
        base_te = np.reshape(base_te, (-1, 1))
        base_tr = np.reshape(base_tr, (-1, 1))
        
        return base_te, base_tr

    def emulate(tensors, ter):
        from tensorflow.keras import backend as K
        from tensorflow.keras.layers import Multiply

        te = ter[:, 0:1]
        tr = ter[:, 1:2]
        pd = tensors[0]
        t1 = tensors[1]
        t2 = tensors[2]

        # Generate signal
        g = pd * (1 - K.exp(tensorflow.math.divide_no_nan(-tr[:, None, None], t1))) * K.exp(tensorflow.math.divide_no_nan(-te[:, None, None], t2))
        # g = pd * (1 - tensorflow.math.pow(t1, tr[:, None, None])) * tensorflow.math.pow(t2, te[:, None, None])
        return g

    def build_discriminator(self, df, img_rows, img_cols, channels): 
        # import tensorflow_addons as tfa
        from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense, Dropout, Embedding, Reshape, Add, Lambda
        from tensorflow.keras.layers import Multiply, GaussianNoise, ReLU, LeakyReLU, Concatenate, MaxPooling2D, concatenate,  LeakyReLU, multiply
        from tensorflow.keras.models import Model
        from tensorflow.keras.initializers import RandomNormal
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.layers import Concatenate
        from tensorflow.keras.layers import BatchNormalization
        tensorflow.random.set_seed(113)
 
        std = self.std
        do = 0.4
        init = RandomNormal(stddev=0.02)
        # source image input
        in_img = Input(shape=(img_rows, img_cols, channels), batch_size=self.batch_size)
        # in_ter = Input(shape=(2), batch_size=self.batch_size)
        # ter = GaussianNoise(0.05)(in_ter)
        # ter = tensorflow.tile(tensorflow.expand_dims(tensorflow.expand_dims(ter, 1), 1), tensorflow.constant([1,256,256,1], tensorflow.int32))
        #d = in_img # Concatenate()([in_img + ter[:, :, :, 0:1], in_img + ter[:, :, :, 1:2]])
        d = GaussianNoise(std)(in_img)
        kernel_size = (3, 3)


        def relu_range(x):
            x = tensorflow.where(tensorflow.keras.backend.greater(x, 0), x, tensorflow.keras.backend.zeros_like(x))
            x = tensorflow.where(tensorflow.keras.backend.less(x, 1), x, tensorflow.keras.backend.ones_like(x))
            return x 

        # d = in_img
        d = GaussianNoise(std)(in_img)
        d = Conv2D(4,  kernel_size, padding='same', kernel_initializer=init)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(16,  kernel_size, padding='same', kernel_initializer=init)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(df,  kernel_size, strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(df*2,  kernel_size, strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = Dropout(do)(d)
        d = GaussianNoise(std)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(df*4,  kernel_size, strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = Dropout(do)(d)
        d = GaussianNoise(std)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(df*4,  kernel_size, strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = Dropout(do)(d)
        d = GaussianNoise(std)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(df*4,  kernel_size, strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = Dropout(do)(d)
        d = GaussianNoise(std)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(6,  kernel_size, padding='same', kernel_initializer=init)(d)
        d_flat = Flatten()(d)

        d = Dense(256)(d_flat)
        d = Dense(128)(d)
        d = Dense(32)(d)
        d = Dense(3)(d)
        d = Lambda(relu_range)(d)

        # # Flatten
        # d = Flatten()(d)
        # d = Dense(6)(d)

        # # Output
        # d = Activation('softmax')(d)
        # define model
        model = Model([in_img], [d[:, 0:1], d[:, 1:2], d[:, 2:3]])
        # compile model
        return model
 
    def build_generator(self): 
        from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense, Concatenate
        from tensorflow.keras.layers import Multiply, GaussianNoise, ReLU, LeakyReLU, UpSampling2D, Lambda, Activation
        from tensorflow.keras.models import Model
        # import tensorflow
        # from tensorflow.keras import backend as K

        def s_range(x):
            x = tensorflow.where(tensorflow.keras.backend.greater(x, 0), x, tensorflow.keras.backend.zeros_like(x))
            x = tensorflow.where(tensorflow.keras.backend.less(x, 1), x, tensorflow.keras.backend.ones_like(x))
            return x 

        def pd_range(x):
            thr = 0
            up_thr = 1
            x = tensorflow.where(tensorflow.keras.backend.greater(x, thr), x, thr * tensorflow.keras.backend.ones_like(x))
            x = tensorflow.where(tensorflow.keras.backend.less(x, up_thr), x, up_thr * tensorflow.keras.backend.ones_like(x))
            return x 

        def t1_range(x):
            thr = 0.001
            up_thr = 10.0
            x = tensorflow.where(tensorflow.keras.backend.greater(x, thr), x, thr * tensorflow.keras.backend.ones_like(x))
            x = tensorflow.where(tensorflow.keras.backend.less(x, up_thr), x, up_thr * tensorflow.keras.backend.ones_like(x))
            return x 

        def t2_range(x):
            thr = 0.001
            up_thr = 5.0
            x = tensorflow.where(tensorflow.keras.backend.greater(x, thr), x, thr * tensorflow.keras.backend.ones_like(x))
            x = tensorflow.where(tensorflow.keras.backend.less(x, up_thr), x, up_thr*tensorflow.keras.backend.ones_like(x))
            return x 

        def normalize(tensor):
            scale = K.max(tensor, axis=(1, 2, 3))
            return tensorflow.math.divide_no_nan(tensor, scale[:, None, None, None])

        def normalize_z(tensor):
            thr = 0.0
            up_thr = 1.0
            tensor = tensorflow.where(tensorflow.keras.backend.greater(tensor, thr), tensor, thr * tensorflow.keras.backend.ones_like(tensor))
            tensor = tensorflow.where(tensorflow.keras.backend.less(tensor, up_thr), tensor, up_thr*tensorflow.keras.backend.ones_like(tensor))
            t_mean = K.mean(tensor, axis=(1, 2, 3))
            t_std = K.std(tensor, axis=(1, 2, 3))
            return tensorflow.math.divide_no_nan(tensor - t_mean[:, None, None, None], t_std[:, None, None, None])
        """U-Net Generator"""

        #Fixed parameters
        kernel_size = 3
        gf = 32


        # Image input
        rows, cols, channels = 256, 256 ,1
        d0 = Input(shape=(rows, cols, channels))
        ter = Input(shape=(2))

        U3 = UNet(gf, kernel_size, (rows, cols, channels))
        PD, T1, T2 = U3(d0)
        PD = Lambda(pd_range, name='pd')(PD)
        T1 = Lambda(t1_range, name='t1')(T1)
        T2 = Lambda(t2_range, name='t2')(T2)

        
        s = MTNN.emulate([PD, T1, T2], ter)
        # s = Lambda(s_range)(s)
        # s = Lambda(normalize)(s)
        s = Lambda(normalize_z)(s)
        
        return Model([d0, ter], s)

    def print_results(self, epoch):
        import tensorflow
        import matplotlib.pyplot as plt
        from tensorflow.keras.utils import to_categorical
        base_ter = np.array([[0.075, 4.5],  [0.12, 4.5] , [0.008, 0.4], [0.008, 0.75], [0.008, 4.5], [0, 0]], dtype=np.float32)
        
        test_generator = Model(inputs=[self.generator.inputs[0]], 
                        outputs=[self.generator.get_layer('pd').output,
                        self.generator.get_layer('t1').output,
                        self.generator.get_layer('t2').output])
    
        # Compile Models
        test_generator.compile(loss='mean_squared_error')   
        # self.discriminator.save(f'{self.save_dir}{epoch}_d.h5')
        self.val_seq.start(workers=4, max_queue_size=5)
        data_seq = self.val_seq.get()
        x, y = next(data_seq)

        for i in range(5):
            x[i] = (x[i] - np.mean(x[i], axis=(1, 2, 3))[:, None, None, None]) / np.std(x[i], axis=(1, 2, 3))[:, None, None, None]

        for idx in range(10):
            pd, t1, t2, mask = y
            mask = mask[idx, :, :, 0]
            pd = pd[idx, :, :, 0, 0]
            t1 = t1[idx, :, :, 0, 0]
            t2 = t2[idx, :, :, 0, 0]
            idx_in = np.random.randint(5)

            comps = np.array(test_generator.predict(x[idx_in]))
            PD = comps[0][idx, :, :, 0]
            T1 = comps[1][idx, :, :, 0]
            T2 = comps[2][idx, :, :, 0]
            

            fig = plt.figure(figsize=(15, 15))
            plt.subplot(4, 3, 7)
            plt.imshow(PD * mask, cmap='gray', interpolation='none')
            plt.axis('off')
            plt.colorbar()
            plt.subplot(4, 3, 8)
            plt.imshow(T1 * mask, cmap='gray', interpolation='none', vmin=np.min(t1 * mask), vmax=np.quantile(t1 * mask, 0.99))
            plt.axis('off')
            plt.colorbar()
            plt.subplot(4, 3, 9)
            plt.imshow(T2 * mask, cmap='gray', interpolation='none', vmin=np.min(t2 * mask), vmax=np.quantile(t2 * mask, 0.99))
            plt.axis('off')
            plt.colorbar()

            plt.subplot(4, 3, 10)
            plt.imshow(pd * mask, cmap='gray', interpolation='none')
            plt.axis('off')
            plt.colorbar()
            plt.subplot(4, 3, 11)
            plt.imshow(t1 * mask, cmap='gray', interpolation='none', vmin=np.min(t1 * mask), vmax=np.quantile(t1 * mask, 0.99))
            plt.axis('off')
            plt.colorbar()
            plt.subplot(4, 3, 12)
            plt.imshow(t2 * mask, cmap='gray', interpolation='none', vmin=np.min(t2 * mask), vmax=np.quantile(t2 * mask, 0.99))
            plt.axis('off')
            plt.colorbar()

            for i in range(5):
                i_map = np.array([i for p in range(self.val_gen.batch_size)])
                out_te, out_tr = self.make_map(i_map)
                out_te_g = np.interp(out_te, (0, 1), (0.008, 0.12))
                out_tr_g = np.interp(out_tr, (0, 1), (0.4, 4.5))
                out_te_int = int(np.round(np.interp(out_te[idx, 0], (0, 1), (8, 120))))
                out_tr_int = int(np.round(np.interp(out_tr[idx, 0], (0, 1), (400, 4500))))

                plt.subplot(4, 5, i + 1)
                discriminator_pred = self.discriminator.predict_on_batch(x[i])
                disc_te = int(np.round(np.interp(discriminator_pred[0][idx, 0], (0, 1), (8, 120))))
                disc_tr = int(np.round(np.interp(discriminator_pred[1][idx, 0], (0, 1), (400, 4500))))

                title = str(out_te_int) + ", " + str(out_tr_int) + "\n" + str(disc_te) + ", " + str(disc_tr)
                if i == idx_in:
                    plt.title(title, fontweight="bold")
                else:
                    plt.title(title)
                plt.imshow(x[i][idx, :, :, 0], cmap='gray', interpolation='none')
                plt.axis('off')

                pred = np.array(self.generator.predict([x[idx_in], np.concatenate((out_te_g, out_tr_g), axis=1)]))
                discriminator_pred = self.discriminator.predict_on_batch(pred)
                disc_te = int(np.round(np.interp(discriminator_pred[0][idx, 0], (0, 1), (8, 120))))
                disc_tr = int(np.round(np.interp(discriminator_pred[1][idx, 0], (0, 1), (400, 4500))))
                
                plt.subplot(4, 5, i + 6)
                plt.title(str(disc_te) + ", " + str(disc_tr))
                plt.imshow(pred[idx, :, :, 0], cmap='gray', interpolation='none')
                plt.axis('off')

            fig.tight_layout()
            if self.comet:
                self.comet.log_figure(figure=plt)
            plt.close('all')
            self.val_seq.stop()

def UNet(filt, kernel_size, input_shape):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    tf.random.set_seed(113)
    init = tf.keras.initializers.Ones()
    input_1 = Input(shape=input_shape, name='input_1')
    
    x = tf.pad(input_1, paddings, 'SYMMETRIC')
    x = Conv2D(4, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = BatchNormalization()(x)
    x_0 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = BatchNormalization()(x)
    x_1 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = BatchNormalization()(x)
    x_2 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = BatchNormalization()(x)
    x_3 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = BatchNormalization()(x)
    x_4 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = BatchNormalization()(x)


    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_4])
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*16, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_3])
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*8, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_2])
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*4, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_1])
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt*2, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x,x_0, input_1])
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = tf.pad(x, paddings, 'SYMMETRIC')
    x = Conv2D(filt, kernel_size, activation = 'relu', padding = 'valid')(x)
    x = Conv2D(3, 1, activation = 'relu', padding = 'same')(x) # , bias_initializer=init, kernel_initializer=init
    pd = x[:, :, :, 0:1]
    t1 = x[:, :, :, 1:2]
    t2 = x[:, :, :, 2:3]

    return Model(inputs=[input_1], outputs=[pd, t1, t2])


if __name__ == "__main__":
    import doctest
    doctest.testmod()