import numpy as np
from math import log10, sqrt 
import time
import sys
import tensorflow
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape, Input
from tensorflow.keras.layers import Conv3D, UpSampling3D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
import argparse
sys.path.append("../")
import MLTK

import matplotlib.pyplot as plt

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

class DCGAN(object):
    def __init__(self, dims=15, dims_small = 9):

        self.dims = dims
        self.dims_small = dims_small
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 10
        dropout = 0.4
        # In: 15 x 15 x 15, depth = 1
        # Out: 1, depth=1
        input_shape = (self.dims, self.dims, self.dims, 1)
        self.D.add(Conv3D(depth*1, (5, 5, 5), strides=1, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv3D(depth*2, (5, 5, 5), strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv3D(depth*4, (5, 5, 5), strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv3D(depth*8, (5, 5, 5), strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 48
        kernel = 3
        # In: 100
        # Out: dim x dim x depth
        input_shape = (self.dims_small, self.dims_small, self.dims_small, 1)
        self.G.add(Conv3D(depth, kernel, strides=1, input_shape=input_shape, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Dropout(dropout))

        self.G.add(Conv3D(depth*2, kernel, strides=1, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Dropout(dropout))

        # # In: dim x dim x depth
        # # Out: 2*dim x 2*dim x depth/2
        # self.G.add(Conv3D(depth*2, kernel, padding='same'))
        # self.G.add(BatchNormalization(momentum=0.9))
        # self.G.add(Activation('relu'))
        # self.G.add(Dropout(dropout))

        self.G.add(UpSampling3D())
        self.G.add(Conv3D(depth*3, 2, padding='valid'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Dropout(dropout))

        self.G.add(Conv3D(depth*2, kernel, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Dropout(dropout))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.G.add(Conv3D(1, 1, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        # optimizer = RMSprop(lr=0.0002, decay=6e-8)
        optimizer = Adam(lr=0.00001, beta_1=0.5)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        # optimizer = RMSprop(lr=0.0001, decay=3e-8)
        optimizer = Adam(lr=0.000005, beta_1=0.5)
        input_shape = (self.dims_small, self.dims_small, self.dims_small, 1)
        inp = Input(shape=input_shape)
        # self.AM = Sequential()
        self.D.trainable = False
        # for layer in self.D.layers:
        #     layer.trainable = False
        x = self.G(inp)
        x = self.D(x)
        self.AM = Model(inp, x)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # self.D.trainable = True
        # for layer in self.D.layers:
        #     layer.trainable = True
        return self.AM

class MNIST_DCGAN(object):
    def __init__(self, data_path, out_path, lr_gen):
        from MLTK.data import DataGenerator
        import os
        self.data_path = data_path
        self.out_path = out_path
        self.dims = 5
        self.dims_small = 3
        self.batch_size = 256

        self.tr_gen = DataGenerator(data_path + 'training',
                                    inputs=[['small', False, 'float32']],
                                    outputs=[['big', False, 'float32']],
                                    batch_size=self.batch_size,
                                    shuffle=True)
        self.val_gen = DataGenerator(data_path + 'validating',
                                    inputs=[['small', False, 'float32']],
                                    outputs=[['big', False, 'float32']],
                                    batch_size=self.batch_size,
                                    shuffle=True)
        self.val_gen_big = DataGenerator(data_path + 'validating_big',
                                    inputs=[['small', False, 'float32']],
                                    outputs=[['big', False, 'float32']],
                                    batch_size=1,
                                    shuffle=False)

        opt_gen = Adam(lr=lr_gen)

        self.DCGAN = DCGAN(self.dims, self.dims_small)
        self.discriminator =  self.DCGAN.discriminator_model()
        self.generator = self.DCGAN.generator()
        self.adversarial = self.DCGAN.adversarial_model()

        self.generator.compile(loss='mean_squared_error', optimizer=opt_gen)

    def train(self, train_steps=2000, save_interval=0):
        label_true = np.array([[1] for p in range(self.batch_size)])
        label_fake = np.array([[0] for p in range(self.batch_size)])
        labels = np.concatenate([label_true, label_fake])
        for i in range(train_steps):
            d_loss_list = []
            d_acc_list = []
            a_loss_list = []
            a_acc_list = []
            g_loss_list = []
            for idx in range(len(self.tr_gen)): # len(self.tr_gen)
                small, big= self.tr_gen[idx]
                small = small[0]
                big = big[0]
                
                g_loss = self.generator.train_on_batch(small, big)
                g_loss_list.append(g_loss)
                # big_fake = self.generator.predict(small)
                # bigs = np.concatenate((big, big_fake))
                # d_loss = self.discriminator.train_on_batch(bigs, labels)
                # d_loss_list.append(d_loss[0])
                # d_acc_list.append(d_loss[1])

                # if (idx % 1 == 0):
                #     a_loss = self.adversarial.train_on_batch(small, label_true)
                #     a_loss_list.append(a_loss[0])
                #     a_acc_list.append(a_loss[1])


            log_mesg = "%d: [G loss: %f]" % (int(100*idx/len(self.tr_gen)), np.mean(g_loss_list))
            # log_mesg = "%d: [D loss: %f, acc: %f]" % (int(100*idx/len(self.tr_gen)), np.round(np.mean(d_loss_list), 3), np.round(np.mean(d_acc_list), 3))
            # log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, np.round(np.mean(a_loss_list), 3), np.round(np.mean(a_acc_list), 3))
            print(log_mesg)
            self.tr_gen.on_epoch_end()

            g_loss_list = []
            for idx in range(len(self.val_gen)): #len(self.val_gen)):
                small, big= self.val_gen[idx]
                small = small[0]
                big = big[0]
                
                g_loss = self.generator.test_on_batch(small, big)
                g_loss_list.append(g_loss)

            log_mesg = "VAL: "
            log_mesg = "%s: [G loss: %f]" % (log_mesg, np.mean(g_loss_list))
            print(log_mesg)
            self.val_gen.on_epoch_end()
            tensorflow.keras.backend.clear_session()

            if (i+1)%save_interval==0:
                self.print_results(i+1)

    def print_results(self, epoch):
        import matplotlib.pyplot as plt
        from tensorflow.keras.utils import to_categorical
        small, big = self.val_gen_big[0]
        small, big = small[0], big[0]
        full_big = np.zeros_like(big[0, :, :, :, 0])
        full_map = np.zeros_like(big[0, :, :, :, 0])
        for i in range(6):
            i_idx = i
            I_idx = i * 2
            for j in range(126):
                j_idx = j
                J_idx = j * 2
                for k in range(126):
                    k_idx = k
                    K_idx = k * 2
                    small_batch = small[0:1, i_idx:i_idx+3, j_idx:j_idx+3, k_idx:k_idx+3, 0:1]
                    if (np.mean(small_batch > 0.1) >= 0.5):
                        full_big[I_idx:I_idx+5, J_idx:J_idx+5, K_idx:K_idx+5] += self.generator.predict(small_batch)[0, :, :, :, 0]
                        full_map[I_idx:I_idx+5, J_idx:J_idx+5, K_idx:K_idx+5] += + 1

        scaled_big = np.nan_to_num(full_big / full_map)
        print("PSNR: " + str(PSNR(scaled_big, big[0, :, :, :, 0])))
        fig = plt.figure(figsize=(50, 3), dpi=100)
        plt.subplot(331)
        plt.imshow(small[0, :, :, 32, 0])
        plt.axis('off')
        plt.subplot(332)
        plt.imshow(small[0, :, :, 64, 0])
        plt.axis('off')
        plt.subplot(333)
        plt.imshow(small[0, :, :, 96, 0])
        plt.axis('off')
        plt.subplot(334)
        plt.imshow(scaled_big[:, :, 64])
        plt.axis('off')
        plt.subplot(335)
        plt.imshow(scaled_big[:, :, 128])
        plt.axis('off')
        plt.subplot(336)
        plt.imshow(scaled_big[:, :, 192])
        plt.axis('off')
        plt.subplot(337)
        plt.imshow(big[0, :, :, 64, 0])
        plt.axis('off')
        plt.subplot(338)
        plt.imshow(big[0, :, :, 128, 0])
        plt.axis('off')
        plt.subplot(339)
        plt.imshow(big[0, :, :, 192, 0])
        plt.axis('off')
        fig.tight_layout()
        plt.savefig(self.out_path + 'prediction_%d.png' % (epoch))
        plt.close('all')

if __name__ == '__main__':
    data_path = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0049_3_5/'
    out_path = '/home/attilasimko/Documents/out/'

    parser = argparse.ArgumentParser(description='Welcome.')
    parser.add_argument("--gpu", default=None, help="Number of GPU to use.")
    parser.add_argument("--lr_gen", default=0.001, help="Generator learning rate.")
    args = parser.parse_args()
    gpu = args.gpu
    lr_gen = float(args.lr_gen)

    if gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use, usually either "0" or "1";
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        data_path = '/home/attila/data/DS0049_3_5/'
        out_path = '/home/attila/out/' + str(gpu) + '/'
    mnist_dcgan = MNIST_DCGAN(data_path, out_path, lr_gen)
    mnist_dcgan.train(train_steps=10000, save_interval=1)