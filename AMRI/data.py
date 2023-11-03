# -*- coding: utf-8 -*-
"""
Contains means to read, generate and handle data.

Created on Tue Oct  3 08:20:52 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import os
import re
import abc
import six
import datetime
import tensorflow

import numpy as np
from six import with_metaclass

import utils

try:
    from collections import Generator
    _HAS_GENERATOR = True
except (ImportError):
    _HAS_GENERATOR = False

try:
    import pydicom
    _HAS_DICOM = True
except (ImportError):
    try:
        import dicom as pydicom
        _HAS_DICOM = True
    except (ImportError):
        _HAS_DICOM = False

class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self,
                 data_path,
                 inputs,
                 outputs,
                 merge=False,
                 batch_size=32,
                 shuffle=True
                 ):

        self.data_path = data_path
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.merge = merge

        if data_path is None:
            raise ValueError('The data path is not defined.')

        if not os.path.isdir(data_path):
            raise ValueError(f'The data path ({repr(data_path)}) is not a directory.')

        self.file_idx = 0
        self.file_list = [self.data_path + '/' + s for s in
                          os.listdir(self.data_path)]
        if ~shuffle:
            self.file_list.sort()
        
        self.on_epoch_end()
        with np.load(self.file_list[0], allow_pickle=True) as npzfile:
            self.out_dims = []
            self.in_dims = []
            self.n_channels = 1
            for i in range(len(self.inputs)):
                if (self.inputs[i][0] == 'ones') | \
                   (self.inputs[i][0] == 'zeros'):
                    self.in_dims.append((self.batch_size,
                                        self.inputs[i][1],
                                        self.inputs[i][1],
                                        self.n_channels))
                else:
                    im = npzfile[self.inputs[i][0]]
                    self.in_dims.append((self.batch_size,
                                        *np.shape(im),
                                        self.n_channels))
            for i in range(len(self.outputs)):
                if (self.outputs[i][0] == 'ones') | \
                   (self.outputs[i][0] == 'zeros'):
                    self.out_dims.append((self.batch_size,
                                         self.outputs[i][1],
                                         self.outputs[i][1],
                                         self.n_channels))
                else:
                    im = npzfile[self.outputs[i][0]]
                    self.out_dims.append((self.batch_size,
                                         *np.shape(im),
                                         self.n_channels))
            npzfile.close()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.file_list)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        self.temp_ID = [self.file_list[k] for k in indexes]

        # Generate data
        i, o = self.__data_generation(self.temp_ID)
        return i, o

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)
    
    #@threadsafe_generator
    def __data_generation(self, temp_list):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        inputs = []
        outputs = []

        for i in range(self.inputs.__len__()):
            inputs.append(np.empty(self.in_dims[i]).astype(self.inputs[i][2]))

        for i in range(self.outputs.__len__()):
            outputs.append(np.empty(self.out_dims[i]).astype(self.outputs[i][2]))

        for i, ID in enumerate(temp_list):
            with np.load(ID, allow_pickle=True) as npzfile:
                for idx in range(len(self.inputs)):
                    if self.inputs[idx][0] == 'ones':
                        x = np.ones((self.inputs[idx][1],
                                     self.inputs[idx][1]))
                    elif self.inputs[idx][0] == 'zeros':
                        x = np.zeros((self.inputs[idx][1],
                                      self.inputs[idx][1]))
                    else:
                        x = npzfile[self.inputs[idx][0]] \
                            .astype(self.inputs[idx][2])
                        if self.inputs[idx][1]:
                            x = x / np.max(x)
                    x = np.expand_dims(x, axis=x.ndim)
                    inputs[idx][i, ] = x

                for idx in range(len(self.outputs)):
                    if self.outputs[idx][0] == 'ones':
                        x = np.ones((self.outputs[idx][1],
                                     self.outputs[idx][1]))
                    elif self.outputs[idx][0] == 'zeros':
                        x = np.zeros((self.outputs[idx][1],
                                      self.outputs[idx][1]))
                    else:
                        x = npzfile[self.outputs[idx][0]] \
                            .astype(self.outputs[idx][2])
                        if self.outputs[idx][1]:
                            x = x / np.max(x)
                    x = np.expand_dims(x, axis=x.ndim)
                    outputs[idx][i, ] = x
                npzfile.close()
            
        if self.merge:
            #merge_inputs = np.concatenate([inputs[0], inputs[1], inputs[2]], axis=3)
            merge_outputs = np.concatenate([outputs[0], outputs[1], outputs[2]], axis=3)
            return inputs, merge_outputs
        return inputs, outputs

if __name__ == "__main__":
    import doctest
    doctest.testmod()
