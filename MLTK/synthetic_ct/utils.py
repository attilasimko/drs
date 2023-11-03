import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import math

class KerasDropoutPrediction(object):
    def __init__(self,model):
        self.f = Model(model.inputs, model.layers[-1].output)

    def predict(self,x, n_iter=5):
        result = []
        for _ in range(n_iter):
            result.append(self.f([x]))#, training=True))
        result = tf.math.reduce_std(tf.stack(result, axis=0), axis=0)
        return result

def get_patients(gen):
    patients = []
    for idx in range(len(gen)):
        patient = gen.file_list[idx].split('/')[-1].split(' ')[0]
        if patients.count(patient) == 0:
            patients.append(patient)
    return patients

def custom_loss(y_true, y_pred):
    dice_scale = 1
    sqrt_scale = 0.05
    bone_scale = 0.1
    air_scale = 0.1
    st_scale = 1
    #difference between true label and predicted label
    error_air = tf.boolean_mask(y_true - y_pred, K.less_equal(y_true, -0.1))
    #square of the error
    sqr_error_air = K.square(error_air)
    abs_error_air = K.abs(error_air)
    #mean of the square of the error
    mean_sqr_error_air = K.mean(sqr_error_air)
    mean_abs_error_air = K.mean(abs_error_air)
    #square root of the mean of the square of the error
    sqrt_mean_sqr_error_air = K.sqrt(mean_sqr_error_air)

    #difference between true label and predicted label
    error_st = tf.boolean_mask(y_true - y_pred, tf.logical_and(K.less_equal(y_true, 0.1), K.greater(y_true, -0.1)))  
    #square of the error
    sqr_error_st = K.square(error_st)
    abs_error_st = K.abs(error_st)
    #mean of the square of the error
    mean_sqr_error_st = K.mean(sqr_error_st)
    mean_abs_error_st = K.mean(abs_error_st)
    #square root of the mean of the square of the error
    sqrt_mean_sqr_error_st = K.sqrt(mean_sqr_error_st)

    #difference between true label and predicted label
    error_bone = tf.boolean_mask(y_true - y_pred, K.greater(y_true, 0.1))
    #square of the error
    sqr_error_bone = K.square(error_bone)
    abs_error_bone = K.abs(error_bone)
    #mean of the square of the error
    mean_sqr_error_bone = K.mean(sqr_error_bone)
    mean_abs_error_bone = K.mean(abs_error_bone)
    #square root of the mean of the square of the error
    sqrt_mean_sqr_error_bone = K.sqrt(mean_sqr_error_bone)

    # Mask out bone
    mask_true = tf.where(K.greater(y_true, 0.1), 1.0, 0.0)
    mask_pred = tf.where(K.greater(y_pred, 0.1), 1.0, 0.0)

    smooth = 0.001
    intersection = K.sum(mask_true * mask_pred, axis=[1, 2, 3])
    union = K.sum(mask_true, axis=[1, 2, 3]) + K.sum(mask_pred, axis=[1, 2, 3])
    dice_score = 1 - K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)

    return mean_abs_error_bone + mean_abs_error_st + mean_abs_error_air
    return bone_scale * (sqrt_scale * sqrt_mean_sqr_error_bone + mean_abs_error_bone) + st_scale * (sqrt_scale * sqrt_mean_sqr_error_st + mean_abs_error_st) + air_scale * (sqrt_scale * sqrt_mean_sqr_error_air + mean_abs_error_air) + dice_scale * dice_score


def weighted_loss(y_true, y_pred):
    alpha = 0
    unc = y_pred[:, :, :, 1:2]
    return tf.reduce_mean(tf.square(y_pred[:, :, :, 0:1] - y_true))# / ((tf.math.floor(unc / tf.constant(math.pi))) + 1)) + alpha * tf.reduce_mean(unc)