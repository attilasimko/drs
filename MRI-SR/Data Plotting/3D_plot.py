import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../")
from MLTK.data import DataGenerator


data_path = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/DS0049/'
tr_gen = DataGenerator(data_path + 'training',
                            inputs=[['small', False, 'float32']],
                            outputs=[['big', False, 'float32']],
                            batch_size=1,
                            shuffle=True)

for i in range(100):
    small, big = tr_gen[i]
    plt.subplot(3,2,1)
    plt.imshow(small[0][0, 0, :, :, 0])
    plt.subplot(3,2,2)
    plt.imshow(big[0][0, 0, :, :, 0])

    plt.subplot(3,2,3)
    plt.imshow(small[0][0, :, 0, :, 0])
    plt.subplot(3,2,4)
    plt.imshow(big[0][0, :, 0, :, 0])

    plt.subplot(3,2,5)
    plt.imshow(small[0][0, :, :, 0, 0])
    plt.subplot(3,2,6)
    plt.imshow(big[0][0, :, :, 0, 0])
    plt.show()