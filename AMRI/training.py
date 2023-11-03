import argparse
import gc
import os
import random
import sys
import time
import warnings
from distutils.command.build import build

import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from tensorflow import TensorSpec, function, io
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import OrderedEnqueuer
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2

from data import DataGenerator
from models import KIKI
from utils import (mse, save_progress, transform_image_to_kspace,
                   transform_kspace_to_image)

lrate = 0.00005
batch_size = 4
case = np.array(["sobel", "baseline", "prewitt", "laplace"])
alpha = np.array([0.4, 0.2, 0.1, 0.3], dtype=np.float)
optimizer = "rmsprop"

split_artefacts = True
if (split_artefacts):
    N_ARTEFACTS = 4
else:
    N_ARTEFACTS = 1

# Paths
data_path = '' # The path to the data created by the scripts in the 'data generation' folder
save_path = '' # Output path for progress files and the final model


physical_devices = tensorflow.config.list_physical_devices('GPU')
tensorflow.config.set_visible_devices(physical_devices[0], 'GPU')

# Generators
gen = DataGenerator(data_path + 'training',
                    inputs=[['clean', False, 'float32'],
                            ['index', False, 'int']],
                    outputs=[['corrupt', False, 'float32']],
                    batch_size=batch_size,
                    shuffle=True)

gen_val = DataGenerator(data_path + 'validating',
                    inputs=[['clean', False, 'float32']],
                    outputs=[['corrupt', False, 'float32']],
                    batch_size=batch_size,
                    shuffle=False)

gen_test_d = DataGenerator(data_path + 'testing',
                    inputs=[['clean', False, 'float32']],
                    outputs=[['downsample', False, 'float32']],
                    batch_size=batch_size,
                    shuffle=False)

gen_test_n = DataGenerator(data_path + 'testing',
                    inputs=[['clean', False, 'float32']],
                    outputs=[['noise', False, 'float32']],
                    batch_size=batch_size,
                    shuffle=False)

gen_test_m = DataGenerator(data_path + 'testing',
                    inputs=[['clean', False, 'float32']],
                    outputs=[['motion', False, 'float32']],
                    batch_size=batch_size,
                    shuffle=False)

gen_test_b = DataGenerator(data_path + 'testing',
                    inputs=[['clean', False, 'float32']],
                    outputs=[['bias', False, 'float32']],
                    batch_size=batch_size,
                    shuffle=False)

gen_test_corrupt = DataGenerator(data_path + 'testing',
                    inputs=[['clean', False, 'float32']],
                    outputs=[['corrupt', False, 'float32']],
                    batch_size=batch_size,
                    shuffle=False)

gen_test_mx = DataGenerator(data_path + 'testing',
                    inputs=[['clean', False, 'float32']],
                    outputs=[['mixed', False, 'float32']],
                    batch_size=batch_size,
                    shuffle=False)

num_workers = 4


inp = Input(shape=(320, 320, 1))
kiki = KIKI(lr=lrate, optimizer=optimizer, case=case, alpha=alpha, num_artefacts=N_ARTEFACTS)

# Extra parameters
loop_num = 1
patience_thr = 10
epoch_thr = 100

fig_num = 8
fig_step = 2595

patience = 0
min_loss = np.inf
e = 0
loop_e = 0

loop = 0


fold = 0
while (fold < 4):
    tensorflow.random.set_seed(1)

    tr_seq = OrderedEnqueuer(gen, use_multiprocessing=False, shuffle=False)
    val_seq = OrderedEnqueuer(gen_val, use_multiprocessing=False, shuffle=False)
    start_time = time.time()

    e_loss_tr = []
    e2_loss_tr = []
    k_loss_tr = []
    e_loss = []
    cycle_loss = []
    k_loss = []
    full_loss = []
    h_loss = []
    ga_loss = []
    l_loss = []
    vif_loss = []
    vif_unit_loss = []
    k1_loss = []
    i1_loss = []
    k2_loss = []
    i2_loss = []

    sec_counter = 0
    tr_seq.start(workers=num_workers, max_queue_size=100)
    data_seq = tr_seq.get()
    rng = int(len(gen))
    for i in range(rng):
        hr, lr = next(data_seq)
        index = hr[1]

        if (split_artefacts):
            index = tensorflow.one_hot(index, 4, axis=1)[:, :, 0]
        else:
            index = 0 * np.ones_like(index)
        
        if (fold == 0):
            pred = lr[0]
            loss = kiki.K1.train_on_batch([pred, index], hr[0])
            e_loss_tr.append(loss)
        elif (fold == 1):
            pred = lr[0]
            if (random.choice([True, False])):
                pred = kiki.K1_full.predict_on_batch(pred)
            loss = kiki.I1.train_on_batch([pred, index], hr[0])
            e_loss_tr.append(loss)
        elif (fold == 2):
            pred = lr[0]
            if (random.choice([True, False])):
                pred = kiki.K1_full.predict_on_batch(pred)
            if (random.choice([True, False])):
                pred = kiki.I1_full.predict_on_batch(pred)
            loss = kiki.K2.train_on_batch([pred, index], hr[0])
            e_loss_tr.append(loss)
        elif (fold == 3):
            pred = lr[0]
            if (random.choice([True, False])):
                pred = kiki.K1_full.predict_on_batch(pred)
            if (random.choice([True, False])):
                pred = kiki.I1_full.predict_on_batch(pred)
            if (random.choice([True, False])):
                pred = kiki.K2_full.predict_on_batch(pred)
            loss = kiki.I2.train_on_batch([pred, index], hr[0])
            e_loss_tr.append(loss)
        else:
            raise Exception("Training should be done.")

    gc.collect()

    gen.on_epoch_end()
    tr_seq.stop()

    print("training: " + str(int(time.time() - start_time)) + "s")
    start_time = time.time()

    val_seq.start(workers=num_workers, max_queue_size=50)
    data_seq = val_seq.get()
    rng = int(len(gen_val))
        
    for i in range(rng):
        hr, lr = next(data_seq)

        k1 = kiki.K1_full.predict_on_batch(lr[0])
        k1_loss.append(mse(hr[0], k1).numpy())

        i1 = kiki.I1_full.predict_on_batch(k1)
        i1_loss.append(mse(hr[0], i1).numpy())
        
        k2 = kiki.K2_full.predict_on_batch(i1)
        k2_loss.append(mse(hr[0], k2).numpy())
        
        i2 = kiki.I2_full.predict_on_batch(k2)
        i2_loss.append(mse(hr[0], i2).numpy())

    gc.collect()
    gen_val.on_epoch_end()
    val_seq.stop()
    
    print("validating: " + str(int(time.time() - start_time)) + "s")
    start_time = time.time()

    print(f"Validation loss: \t{str(round(np.mean(k1_loss), 4))} \t{str(round(np.mean(i1_loss), 4))} \t{str(round(np.mean(k2_loss), 4))} \t{str(round(np.mean(i2_loss), 4))}")


    if (fold == 0):
        val_loss = k1_loss
    elif (fold == 1):
        val_loss = i1_loss
    elif (fold == 2):
        val_loss = k2_loss
    elif (fold == 3):
        val_loss = i2_loss

    if (np.mean(val_loss) < min_loss):
        min_loss = np.mean(val_loss)
        patience = 0
        
        kiki.KIKI.save_weights(save_path + "kiki_best.h5")

        ind = 0
        for i in range(0, fig_num*fig_step, fig_step):
            hr, lr = gen_test_d[i]
            save_progress(lr, hr, kiki, save_path, e, None, f"downsample_{i}")
            ind += 1

        for i in range(0, fig_num*fig_step, fig_step):
            hr, lr = gen_test_n[i]
            save_progress(lr, hr, kiki, save_path, e, None, f"noise_{i}")
            ind += 1

        for i in range(0, fig_num*fig_step, fig_step):
            hr, lr = gen_test_m[i]
            save_progress(lr, hr, kiki, save_path, e, None, f"motion_{i}")
            ind += 1

        for i in range(0, fig_num*fig_step, fig_step):
            hr, lr = gen_test_b[i]
            save_progress(lr, hr, kiki, save_path, e, None, f"bias_{i}", True)
            ind += 1

        exp_kiki = Model(inputs=[kiki.KIKI.inputs[0]], 
                        outputs=[kiki.KIKI.outputs[0]])
        full_model = function(lambda x: kiki.KIKI(x)) 
        full_model = full_model.get_concrete_function(TensorSpec(kiki.KIKI.inputs[0].shape, kiki.KIKI.inputs[0].dtype))
        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=save_path,
                        name=f'MICE_version.pb',
                        as_text=False)
        gc.collect()
    else:
        patience += 1

    if ((patience >= patience_thr) | (loop_e >= epoch_thr)):

        kiki.KIKI.load_weights(save_path + "kiki_best.h5")

        loop_e = 0
        fold += 1
        min_loss = np.inf
        patience = 0

    e += 1
    loop_e += 1
    gc.collect()
    tensorflow.keras.backend.clear_session()
    del hr, lr

kiki.KIKI.save_weights(save_path + f"kiki_final.h5")
print("Training is done.")