# -*- coding: utf-8 -*-
"""
Contains helper functions and utility functions for use in the library.

Created on Mon Oct  9 14:05:26 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
__all__ = ["running_mean", "mse", "save_state", "Timer", "ReflectPadding2D", "MyCustomWeightShifter"]

import tensorflow
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import gc
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.constraints import Constraint
import logging
import tensorflow as tf
# from typeguard import typechecked

# @tf.keras.utils.register_keras_serializable(package="Addons")
class WeightNormalization(tf.keras.layers.Wrapper):
    """Performs weight normalization.
    This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction.
    This speeds up convergence by improving the
    conditioning of the optimization problem.
    See [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868).
    Wrap `tf.keras.layers.Conv2D`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> conv2d = WeightNormalization(tf.keras.layers.Conv2D(2, 2), data_init=False)
    >>> y = conv2d(x)
    >>> y.shape
    TensorShape([1, 9, 9, 2])
    Wrap `tf.keras.layers.Dense`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> dense = WeightNormalization(tf.keras.layers.Dense(10), data_init=False)
    >>> y = dense(x)
    >>> y.shape
    TensorShape([1, 10, 10, 10])
    Arguments:
      layer: A `tf.keras.layers.Layer` instance.
      data_init: If `True` use data dependent variable initialization.
    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights.
      NotImplementedError: If `data_init` is True and running graph execution.
    """

    # @typechecked
    def __init__(self, layer: tf.keras.layers, data_init: bool = True, **kwargs):
        super().__init__(layer, **kwargs)
        self.data_init = data_init
        self._track_trackable(layer, name="layer")
        self.is_rnn = isinstance(self.layer, tf.keras.layers.RNN)

        if self.data_init and self.is_rnn:
            logging.warning(
                "WeightNormalization: Using `data_init=True` with RNNs "
                "is advised against by the paper. Use `data_init=False`."
            )

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer.cell if self.is_rnn else self.layer

        if not hasattr(kernel_layer, "kernel"):
            raise ValueError(
                "`WeightNormalization` must wrap a layer that"
                " contains a `kernel` for weights"
            )

        if self.is_rnn:
            kernel = kernel_layer.recurrent_kernel
        else:
            kernel = kernel_layer.kernel

        # The kernel's filter or unit dimension is -1
        self.layer_depth = int(kernel.shape[-1])
        self.kernel_norm_axes = list(range(kernel.shape.rank - 1))

        self.g = self.add_weight(
            name="g",
            shape=(self.layer_depth,),
            initializer="ones",
            dtype=kernel.dtype,
            trainable=True,
        )
        self.v = kernel

        self._initialized = self.add_weight(
            name="initialized",
            shape=None,
            initializer="zeros",
            dtype=tf.dtypes.bool,
            trainable=False,
        )

        if self.data_init:
            # Used for data initialization in self._data_dep_init.
            with tf.name_scope("data_dep_init"):
                layer_config = tf.keras.layers.serialize(self.layer)
                layer_config["config"]["trainable"] = False
                self._naked_clone_layer = tf.keras.layers.deserialize(layer_config)
                self._naked_clone_layer.build(input_shape)
                self._naked_clone_layer.set_weights(self.layer.get_weights())
                if not self.is_rnn:
                    self._naked_clone_layer.activation = None

        self.built = True

    def call(self, inputs):
        """Call `Layer`"""

        def _do_nothing():
            return tf.identity(self.g)

        def _update_weights():
            # Ensure we read `self.g` after _update_weights.
            with tf.control_dependencies(self._initialize_weights(inputs)):
                return tf.identity(self.g)

        g = tf.cond(self._initialized, _do_nothing, _update_weights)

        with tf.name_scope("compute_weights"):
            # Replace kernel by normalized weight variable.
            kernel = tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * g

            if self.is_rnn:
                self.layer.cell.recurrent_kernel = kernel
                update_kernel = tf.identity(self.layer.cell.recurrent_kernel)
            else:
                self.layer.kernel = kernel
                update_kernel = tf.identity(self.layer.kernel)

            # Ensure we calculate result after updating kernel.
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    def _initialize_weights(self, inputs):
        """Initialize weight g.
        The initial value of g could either from the initial value in v,
        or by the input value if self.data_init is True.
        """
        with tf.control_dependencies(
            [
                tf.debugging.assert_equal(  # pylint: disable=bad-continuation
                    self._initialized, False, message="The layer has been initialized."
                )
            ]
        ):
            if self.data_init:
                assign_tensors = self._data_dep_init(inputs)
            else:
                assign_tensors = self._init_norm()
            assign_tensors.append(self._initialized.assign(True))
            return assign_tensors

    def _init_norm(self):
        """Set the weight g with the norm of the weight vector."""
        with tf.name_scope("init_norm"):
            v_flat = tf.reshape(self.v, [-1, self.layer_depth])
            v_norm = tf.linalg.norm(v_flat, axis=0)
            g_tensor = self.g.assign(tf.reshape(v_norm, (self.layer_depth,)))
            return [g_tensor]

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""
        with tf.name_scope("data_dep_init"):
            # Generate data dependent init values
            x_init = self._naked_clone_layer(inputs)
            data_norm_axes = list(range(x_init.shape.rank - 1))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1.0 / tf.math.sqrt(v_init + 1e-10)

            # RNNs have fused kernels that are tiled
            # Repeat scale_init to match the shape of fused kernel
            # Note: This is only to support the operation,
            # the paper advises against RNN+data_dep_init
            if scale_init.shape[0] != self.g.shape[0]:
                rep = int(self.g.shape[0] / scale_init.shape[0])
                scale_init = tf.tile(scale_init, [rep])

            # Assign data dependent init values
            g_tensor = self.g.assign(self.g * scale_init)
            if hasattr(self.layer, "bias") and self.layer.bias is not None:
                bias_tensor = self.layer.bias.assign(-m_init * scale_init)
                return [g_tensor, bias_tensor]
            else:
                return [g_tensor]

    def get_config(self):
        config = {"data_init": self.data_init}
        base_config = super().get_config()
        return {**base_config, **config}

    def remove(self):
        kernel = tf.Variable(
            tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * self.g,
            name="recurrent_kernel" if self.is_rnn else "kernel",
        )

        if self.is_rnn:
            self.layer.cell.recurrent_kernel = kernel
        else:
            self.layer.kernel = kernel

        return self.layer

def BFC_eval_cv(img, mask): # 3:GM, 4:WM, 8:Skull
    cv_csf = BFC_cv(img, mask == 2)

    cv_gm = BFC_cv(img, mask == 3)

    cv_wm = BFC_cv(img, mask == 4)

    return [cv_gm, cv_wm, cv_csf]

def BFC_cv(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    cv = []
    ind = [slice(None)]*(values.ndim)
    for i in range(np.shape(values)[-4]):
        ind[-4] = i
        if np.sum(weights[ind]) >= 256*256*0.05:
            average = np.average(values[ind], weights=weights[ind])
            variance = np.average((values[ind] - average)**2, weights=weights[ind])
            cv.append(np.sqrt(variance) / average)
            
    return np.mean(cv)


class _Merge(Layer):
  """Generic merge layer for elementwise merge functions.
  Used to implement `Sum`, `Average`, etc.
  """

  def __init__(self, **kwargs):
    """Intializes a Merge layer.
    Arguments:
      **kwargs: standard layer keyword arguments.
    """
    super(_Merge, self).__init__(**kwargs)
    self.supports_masking = True

  def _merge_function(self, inputs):
    raise NotImplementedError

  def _compute_elemwise_op_output_shape(self, shape1, shape2):
    """Computes the shape of the resultant of an elementwise operation.
    Arguments:
        shape1: tuple or None. Shape of the first tensor
        shape2: tuple or None. Shape of the second tensor
    Returns:
        expected output shape when an element-wise operation is
        carried out on 2 tensors with shapes shape1 and shape2.
        tuple or None.
    Raises:
        ValueError: if shape1 and shape2 are not compatible for
            element-wise operations.
    """
    if None in [shape1, shape2]:
      return None
    elif len(shape1) < len(shape2):
      return self._compute_elemwise_op_output_shape(shape2, shape1)
    elif not shape2:
      return shape1
    output_shape = list(shape1[:-len(shape2)])
    for i, j in zip(shape1[-len(shape2):], shape2):
      if i is None or j is None:
        output_shape.append(None)
      elif i == 1:
        output_shape.append(j)
      elif j == 1:
        output_shape.append(i)
      else:
        if i != j:
          raise ValueError(
              'Operands could not be broadcast '
              'together with shapes ' + str(shape1) + ' ' + str(shape2))
        output_shape.append(i)
    return tuple(output_shape)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # Used purely for shape validation.
    if not isinstance(input_shape[0], tuple):
      raise ValueError('A merge layer should be called on a list of inputs.')
    if len(input_shape) < 2:
      raise ValueError('A merge layer should be called '
                       'on a list of at least 2 inputs. '
                       'Got ' + str(len(input_shape)) + ' inputs.')
    batch_sizes = {s[0] for s in input_shape if s} - {None}
    if len(batch_sizes) > 1:
      raise ValueError(
          'Can not merge tensors with different '
          'batch sizes. Got tensors with shapes : ' + str(input_shape))
    if input_shape[0] is None:
      output_shape = None
    else:
      output_shape = input_shape[0][1:]
    for i in range(1, len(input_shape)):
      if input_shape[i] is None:
        shape = None
      else:
        shape = input_shape[i][1:]
      output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
    # If the inputs have different ranks, we have to reshape them
    # to make them broadcastable.
    if None not in input_shape and len(set(map(len, input_shape))) == 1:
      self._reshape_required = False
    else:
      self._reshape_required = True

  def call(self, inputs):
    if not isinstance(inputs, (list, tuple)):
      raise ValueError('A merge layer should be called on a list of inputs.')
    if self._reshape_required:
      reshaped_inputs = []
      input_ndims = list(map(K.ndim, inputs))
      if None not in input_ndims:
        # If ranks of all inputs are available,
        # we simply expand each of them at axis=1
        # until all of them have the same rank.
        max_ndim = max(input_ndims)
        for x in inputs:
          x_ndim = K.ndim(x)
          for _ in range(max_ndim - x_ndim):
            x = array_ops.expand_dims(x, axis=1)
          reshaped_inputs.append(x)
        return self._merge_function(reshaped_inputs)
      else:
        # Transpose all inputs so that batch size is the last dimension.
        # (batch_size, dim1, dim2, ... ) -> (dim1, dim2, ... , batch_size)
        transposed = False
        for x in inputs:
          x_ndim = K.ndim(x)
          if x_ndim is None:
            x_shape = array_ops.shape(x)
            batch_size = x_shape[0]
            new_shape = K.concatenate(
                [x_shape[1:],
                 array_ops.expand_dims(batch_size, axis=-1)])
            x_transposed = array_ops.reshape(
                x,
                array_ops.stack(
                    [batch_size, math_ops.reduce_prod(x_shape[1:])], axis=0))
            x_transposed = array_ops.transpose(x_transposed, perm=(1, 0))
            x_transposed = array_ops.reshape(x_transposed, new_shape)
            reshaped_inputs.append(x_transposed)
            transposed = True
          elif x_ndim > 1:
            dims = list(range(1, x_ndim)) + [0]
            reshaped_inputs.append(array_ops.transpose(x, perm=dims))
            transposed = True
          else:
            # We don't transpose inputs if they are 1D vectors or scalars.
            reshaped_inputs.append(x)
        y = self._merge_function(reshaped_inputs)
        y_ndim = K.ndim(y)
        if transposed:
          # If inputs have been transposed, we have to transpose the output too.
          if y_ndim is None:
            y_shape = array_ops.shape(y)
            y_ndim = array_ops.shape(y_shape)[0]
            batch_size = y_shape[y_ndim - 1]
            new_shape = K.concatenate([
                array_ops.expand_dims(batch_size, axis=-1), y_shape[:y_ndim - 1]
            ])
            y = array_ops.reshape(y, (-1, batch_size))
            y = array_ops.transpose(y, perm=(1, 0))
            y = array_ops.reshape(y, new_shape)
          elif y_ndim > 1:
            dims = [y_ndim - 1] + list(range(y_ndim - 1))
            y = array_ops.transpose(y, perm=dims)
        return y
    else:
      return self._merge_function(inputs)

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if input_shape[0] is None:
      output_shape = None
    else:
      output_shape = input_shape[0][1:]
    for i in range(1, len(input_shape)):
      if input_shape[i] is None:
        shape = None
      else:
        shape = input_shape[i][1:]
      output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
    batch_sizes = {s[0] for s in input_shape if s is not None} - {None}
    if len(batch_sizes) == 1:
      output_shape = (list(batch_sizes)[0],) + output_shape
    else:
      output_shape = (None,) + output_shape
    return output_shape

  def compute_mask(self, inputs, mask=None):
    if mask is None:
      return None
    if not isinstance(mask, (tuple, list)):
      raise ValueError('`mask` should be a list.')
    if not isinstance(inputs, (tuple, list)):
      raise ValueError('`inputs` should be a list.')
    if len(mask) != len(inputs):
      raise ValueError('The lists `inputs` and `mask` '
                       'should have the same length.')
    if all(m is None for m in mask):
      return None
    masks = [array_ops.expand_dims(m, axis=0) for m in mask if m is not None]
    return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        in_shape = K.shape(inputs[0])
        shape = K.concatenate([in_shape[0:1], K.ones_like(in_shape[1:], dtype='int32')], axis=0)
        alpha = K.random_uniform(shape)
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

def ConvertToPB(model, outPath):
    import os
    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="InputLayer"))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=outPath,
                    name='QM.pb',
                    as_text=False)
    return None


def gradient_penalty(discriminator, generator):
    import tensorflow as tf
    def loss(y_true, y_pred):
        batch_img = batch_x[0]
        batch_te = batch_x[1]
        batch_tr = batch_x[2]

        # [b, h, w, c]
        t = tf.random.uniform([batch_img.shape[0], 1, 1, 1])
        # [b, 1, 1, 1] => [b, h, w, c]
        t = tf.broadcast_to(t, batch_img.shape)

        interpolate = t * batch_img + (1 - t) * fake_image

        with tf.GradientTape() as tape:
            tape.watch([interpolate])
            d_interpolate_logits = discriminator([interpolate, batch_te, batch_tr], training=True)
        grads = tape.gradient(d_interpolate_logits, interpolate)

        # grads:[b, h, w, c] => [b, -1]
        grads = tf.reshape(grads, [grads.shape[0], -1])
        gp = tf.norm(grads, axis=1)  # [b]
        gp = tf.reduce_mean((gp - 1) ** 2)

        return gp
    return loss

def gradient_penalty_loss(y_true, y_pred, averaged_samples, discriminator):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors

    with tensorflow.GradientTape() as tape:
        tape.watch(y_pred)
        disc_ave = discriminator(y_pred)
        gradients = tape.gradient(disc_ave, y_pred)
    

    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                            axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return K.mean(y_true * y_pred)

class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value

	# clip model weights to hypercube
	def __call__(self, weights):
		return K.clip(weights, -self.clip_value, self.clip_value)

	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}


def smooth(labels):
    return labels * 0.8 + 0.1
    

def truediv(tensor):
    from tensorflow.keras import backend as K
    a = tensor[0]
    b = tensor[1]
    return tensorflow.math.divide_no_nan(a, b)

def normalize_img(tensor):
    from tensorflow.keras import backend as K
    scale = K.max(tensor, axis=(1, 2, 3))
    tensor = tensorflow.math.divide_no_nan(tensor, scale[:, None, None, None])
    return tensor

def running_mean(x, N, k):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    ave = (cumsum[N:] - cumsum[:-N]) / float(N)
    if len(ave) > k-1:
        dif = x[-k:] - ave[-k:]
        return all(dif[-k:] > 0)
    else:
        return False

def mse(A, B):
    return (np.abs(A - B)).mean(axis=None)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


    
def save_state(img, save_path, epoch, model=None, save_interval=0, comet=None):
    matplotlib.use('Agg')
    if save_interval > 0:
        if (epoch+1) % save_interval == 0:
            plt.figure(figsize=(10, 10))
            if model:
                plt.title("Ground truth")
                img = model.predict_on_batch(img)
            else:
                plt.title("Predictions")
                img = img
            for i in range(16):
                plt.subplot(4, 4, i+1)
                image = img[i, :, :, 0]
                image = image / np.mean(image) # Just for me.
                plt.imshow(image)
                plt.colorbar()
                plt.axis('off')
            plt.tight_layout()
            comet.log_figure(figure=plt)
            plt.close('all')
    

class Timer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("--- Elapsed time: %s ---" % self.elapsed(time.time() - self.start_time))


class ReflectPadding2D(tensorflow.keras.layers.Layer):
    """Reflection-padding layer for 2D input (e.g. picture).

    This layer adds rows and columns of reflected versions of the input at the
    top, bottom, left and right side of an image tensor.

    Parameters
    ----------
    padding : int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
        If int, the same symmetric padding is applied to width and height. If
        tuple of 2 ints, interpreted as two different symmetric padding values
        for height and width: `(symmetric_height_pad, symmetric_width_pad)`. If
        tuple of 2 tuples of 2 ints: interpreted as `((top_pad, bottom_pad),
        (left_pad, right_pad))`

    data_format : str
        One of `channels_last` (default) or `channels_first`. The ordering of
        the dimensions in the inputs. `channels_last` corresponds to inputs
        with shape `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`. It
        defaults to the `image_data_format` value found in your Keras config
        file at `~/.keras/keras.json`. If you never set it, then it will be
        "channels_last".

    Examples
    --------
    >>> import nethin.padding as padding
    >>> from keras.layers import Input
    >>> from keras.models import Model
    >>> from keras import optimizers
    >>> import numpy as np
    >>>
    >>> A = np.arange(12).reshape(3, 4).astype(np.float32)
    >>>
    >>> inputs = Input(shape=(3, 4, 1))
    >>> x = neural.ReflectPadding2D(padding=2, data_format="channels_last")(inputs)
    >>> model = Model(inputs=inputs, outputs=x)
    >>> model.predict(A.reshape(1, 3, 4, 1)).reshape(7, 8)
    array([[ 10.,   9.,   8.,   9.,  10.,  11.,  10.,   9.],
           [  6.,   5.,   4.,   5.,   6.,   7.,   6.,   5.],
           [  2.,   1.,   0.,   1.,   2.,   3.,   2.,   1.],
           [  6.,   5.,   4.,   5.,   6.,   7.,   6.,   5.],
           [ 10.,   9.,   8.,   9.,  10.,  11.,  10.,   9.],
           [  6.,   5.,   4.,   5.,   6.,   7.,   6.,   5.],
           [  2.,   1.,   0.,   1.,   2.,   3.,   2.,   1.]], dtype=float32)
    >>>
    >>> inputs = Input(shape=(1, 3, 4))
    >>> x = neural.ReflectPadding2D(padding=1, data_format="channels_first")(inputs)
    >>> model = Model(inputs=inputs, outputs=x)
    >>> model.predict(A.reshape(1, 1, 3, 4)).reshape(5, 6)
    array([[[[  5.,   4.,   5.,   6.,   7.,   6.],
             [  1.,   0.,   1.,   2.,   3.,   2.],
             [  5.,   4.,   5.,   6.,   7.,   6.],
             [  9.,   8.,   9.,  10.,  11.,  10.],
             [  5.,   4.,   5.,   6.,   7.,   6.]]]], dtype=float32)
    """
    def __init__(self, padding=(1, 1), data_format=None, **kwargs):

        super(ReflectPadding2D, self).__init__(**kwargs)

        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        "1st entry of padding")
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       "2nd entry of padding")
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))

        self.data_format = conv_utils.normalize_data_format(data_format)

        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):

        super(ReflectPadding2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        if self.data_format == "channels_last":

            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None

            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None

            return (input_shape[0], rows, cols, input_shape[3])

        elif self.data_format == "channels_first":

            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None

            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None

            return (input_shape[0], input_shape[1], rows, cols)

    def get_config(self):

        config = {"padding": self.padding,
                  "data_format": self.data_format}
        base_config = super(ReflectPadding2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """Performs the actual padding.

        Parameters
        ----------
        inputs : Tensor, rank 4
            4D tensor with shape:
                - If `data_format` is `"channels_last"`:
                    `(batch, rows, cols, channels)`
                - If `data_format` is `"channels_first"`:
                    `(batch, channels, rows, cols)`

        Returns
        -------
        outputs : Tensor, rank 4
            4D tensor with shape:
                - If `data_format` is `"channels_last"`:
                    `(batch, padded_rows, padded_cols, channels)`
                - If `data_format` is `"channels_first"`:
                    `(batch, channels, padded_rows, padded_cols)`
        """
        outputs = K.spatial_2d_padding(inputs,
                                       padding=self.padding,
                                       data_format=self.data_format)

        p00, p01 = self.padding[0][0], self.padding[0][1]
        p10, p11 = self.padding[1][0], self.padding[1][1]
        if self.data_format == "channels_last":

            row0 = K.concatenate([inputs[:, p00:0:-1, p10:0:-1, :],
                                  inputs[:, p00:0:-1, :, :],
                                  inputs[:, p00:0:-1, -2:-2-p11:-1, :]],
                                 axis=2)
            row1 = K.concatenate([inputs[:, :, p10:0:-1, :],
                                  inputs,
                                  inputs[:, :, -2:-2-p11:-1, :]],
                                 axis=2)
            row2 = K.concatenate([inputs[:, -2:-2-p01:-1, p10:0:-1, :],
                                  inputs[:, -2:-2-p01:-1, :, :],
                                  inputs[:, -2:-2-p01:-1, -2:-2-p11:-1, :]],
                                 axis=2)

            outputs = K.concatenate([row0, row1, row2], axis=1)

        else:  # self.data_format == "channels_first"

            row0 = K.concatenate([inputs[:, :, p00:0:-1, p10:0:-1],
                                  inputs[:, :, p00:0:-1, :],
                                  inputs[:, :, p00:0:-1, -2:-2-p11:-1]],
                                 axis=3)
            row1 = K.concatenate([inputs[:, :, :, p10:0:-1],
                                  inputs,
                                  inputs[:, :, :, -2:-2-p11:-1]],
                                 axis=3)
            row2 = K.concatenate([inputs[:, :, -2:-2-p01:-1, p10:0:-1],
                                  inputs[:, :, -2:-2-p01:-1, :],
                                  inputs[:, :, -2:-2-p01:-1, -2:-2-p11:-1]],
                                 axis=3)

            outputs = K.concatenate([row0, row1, row2], axis=2)

        return outputs

class MyCustomWeightShifter(tensorflow.keras.callbacks.Callback):
    """Shift weight towards regularization when a metric has stopped 
    improving.
    # Example
    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```
    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the regularization weight will be 
        increased. new_lr = lr * factor
        patience: number of epochs that produced the monitored
            quantity with no improvement after which training will
            be stopped.
            Validation quantities may not be produced for every
            epoch, if the validation frequency
            (`model.fit(validation_freq=5)`) is greater than one.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        min_delta: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        max_lr: upper bound on the learning rate.
    """

    def __init__(self, monitor='val_loss', factor=2, patience=5,
                 verbose=0, mode='auto', min_delta=1e-4, cooldown=0, max_lr=0.5,
                 **kwargs):

        self.monitor = monitor
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            warnings.warn('`epsilon` argument is deprecated and '
                          'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.max_lr = max_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = self.model.loss_weights[1]
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(self.model.loss_weights[1])
                    if old_lr < self.max_lr:
                        new_lr = old_lr * self.factor
                        new_lr = min(new_lr, self.max_lr)
                        tensorflow.keras.backend.set_value(self.model.loss_weights, [1, new_lr, new_lr])
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing '
                                  'learning rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
        print(self.model.loss_weights)

    def in_cooldown(self):
        return self.cooldown_counter > 0



def eval_ssim(img, gt, mask): # 3:GM, 4:WM, 8:Skull
    from skimage.measure import compare_ssim as ssim
    thr = 256 * 256 * 0.05

    if sum(sum(mask)) >= thr:
        ssim_tissue = ssim(img[mask], gt[mask])
    else:
        ssim_tissue = np.nan

    return ssim_tissue

def c_variance(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    thr = 256 * 256 * 0.05
    if sum(sum(weights)) >= thr:
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values - average)**2, weights=weights)
        cv = np.sqrt(variance) / average
    else:
        cv = np.nan
    return cv

def QuickVal(model, gen, comet):
    from skimage.measure import compare_nrmse
    import gc 
    hist = []
    cv_gm_hist = []
    cv_wm_hist=[]
    cv_csf_hist=[]
    ssim_gm_hist = []
    ssim_wm_hist=[]
    ssim_csf_hist=[]
    reg_list = []
    for gen_idx in range(gen.__len__()):
        inputImage, inputBias = gen.__getitem__(gen_idx)
        inputImage = inputImage[0]
        bias = model.predict_on_batch(inputImage)
        correctedImage = np.divide(inputImage, bias, out=np.zeros_like(bias), where=bias!=0)
        reg_bias = model.predict_on_batch(correctedImage)
        reg = np.divide(correctedImage, reg_bias, out=np.zeros_like(reg_bias), where=reg_bias!=0)
        tensorflow.keras.backend.clear_session()
        for tissue_idx in range(3):
            tss = int(tissue_idx + 2)
            for idx in range(gen.batch_size):
                tissue = inputBias[1][idx, :, :, 0] == tss  # 2: CSF, 3:GM, 4:WM
                if sum(sum(tissue)) >= 256 * 256 * 0.05:
                    inputSlice = inputImage[idx, :, :, 0]
                    #inputSlice = np.interp(inputSlice, (inputSlice.min(), inputSlice.max()),
                    #                    (0, 1))
                    # print(np.mean(bias_pred))
                    bias_pred = bias[idx, :, :, 0]
                    outputSlice = np.divide(inputSlice, bias_pred, out=np.zeros_like(bias_pred), where=bias_pred!=0)
                    origImage = inputBias[0][idx, :, :, 0]
                    origImage = np.divide(inputSlice, origImage, out=np.zeros_like(origImage), where=origImage!=0)

                    orig_hist = np.interp(origImage, (np.min(origImage),
                                                    np.max(origImage)),
                                        (0, 1))
                    # input_hist = np.interp(inputSlice, (np.min(inputSlice),
                    #                                 np.max(inputSlice)),
                    #                     (0, 1))
                    output_hist = np.interp(outputSlice, (np.min(outputSlice),
                                                    np.max(outputSlice)),
                                        (0, 1))
                    reg_slice = reg[idx, :, :, 0]
                    reg_hist = np.interp(reg_slice, (np.min(reg_slice), np.max(reg_slice)), (0, 1))

                    reg_list.append(compare_nrmse(output_hist, reg_hist))
                    cv = c_variance(output_hist, tissue)
                    ssim = eval_ssim(output_hist, orig_hist, tissue)
                    if tss == 2:
                        cv_csf_hist.append(cv)
                        ssim_csf_hist.append(ssim)
                    elif tss == 3:
                        cv_gm_hist.append(cv)
                        ssim_gm_hist.append(ssim)
                    elif tss == 4:
                        cv_wm_hist.append(cv)
                        ssim_wm_hist.append(ssim)

                        
    gc.collect()
    comet.log_metrics( {'cv_csf' : np.nanmean(cv_csf_hist),
                        'cv_gm' : np.nanmean(cv_gm_hist),
                        'cv_wm' : np.nanmean(cv_wm_hist),
                        'ssim_csf' : np.nanmean(ssim_csf_hist),
                        'ssim_gm' : np.nanmean(ssim_gm_hist),
                        'ssim_wm' : np.nanmean(ssim_wm_hist),
                        'reg': np.nanmean(reg_list)} )
    
    return np.nanmean(ssim_csf_hist) + np.nanmean(ssim_gm_hist) + np.nanmean(ssim_wm_hist)

class MyCustomCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, corrector, generator, save_path, comet):
        self.corrector = corrector
        self.generator = generator
        self.save_path = save_path
        self.comet = comet
        self.monitor = 0


    def on_epoch_end(self, epoch, logs=None):
        if self.comet:
            self.comet.set_step(epoch)
        hist = QuickVal(self.corrector, self.generator, self.comet)
        logs.__setitem__("monitor", hist)
        sample, trash = self.generator.__getitem__(0)
        save_state(img=sample[0], model=self.corrector,
                    save_interval=1, epoch=epoch, save_path=self.save_path, comet = self.comet)
        save_state(img=trash[0],
                    save_interval=1, epoch=epoch, save_path=self.save_path, comet = self.comet)
        tensorflow.keras.backend.clear_session()
        gc.collect()


class ReflectionPadding2D(tensorflow.keras.layers.Layer):
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'padding': self.padding,
            'input_spec': self.input_spec
        })
        return config

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

def hcomp(oImage, oTissue, figureID=2, title=''):
    """
    The tissue maps:  [Background, CSF, Gray Matter, White Matter, Fat,
                       Muscle, Muscle/Skin, Skull, Vessels, Around Fat,
                       Dura Matter, Bone Marrow]

    Selected tissues: [            CSF, Gray Matter, White Matter, Fat,
                       Muscle, Muscle/Skin, Skull,
                       Dura Matter, Bone Marrow]
    """
    titles = ['Background', 'CSF', 'Gray Matter', 'White Matter', 'Fat',
              'Muscle', 'Muscle/Skin', 'Skull', 'Vessels', 'Around Fat',
              'Dura Matter', 'Bone Marrow']
    tmap = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    from scipy.stats import norm
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    Image = oImage.flatten()
    Tissue = oTissue.flatten()
    stats = []

    plt.figure(figureID)
    for imID in range(1, len(tmap)):
        plt.subplot(4, 3, imID+1)
        data = Image[Tissue == tmap[imID]]
        if len(data) > 500:
            (mu, sigma) = norm.fit(data)
            k2, p = scipy.stats.normaltest(data)

            # the histogram of the data
            n, bins, patches = plt.hist(data, 25, normed=1,
                                        facecolor='green', alpha=0.75)

            # add a 'best fit' line
            y = mlab.normpdf(bins, mu, sigma)
            l = plt.plot(bins, y, 'r--', linewidth=2)

            # plot
            plt.xlabel('Smarts')
            plt.ylabel('Probability')
            plt.title('P-value: ' + str(p))
            plt.grid(True)
    return stats

def new_divide(tensor):
    import tensorflow
    from tensorflow.keras import backend as K
    scale = K.mean(tensor, axis=(1, 2, 3))
    return tensorflow.math.divide_no_nan(tensor, scale[:, None, None, None])


class MyBFCCustomCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, model, val_gen, save_path):
        self.save_path = save_path
        self.model = model
        self.gen = val_gen

    def on_epoch_end(self, epoch, logs=None):
        hist = self.model.evaluate_generator(self.gen)
        print("Validation loss: " + str(hist))
        matplotlib.use('Agg')
        # First
        filename = self.save_path + "/orig_%d.png" % (epoch+1)
        img, data = self.gen[0]
        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            image = img[0][i, :, :, 0]
            #image = tensorflow.signal.idct(image, n=256, norm="ortho")
            #image = tensorflow.transpose(image, [1, 0])
            #image = tensorflow.signal.idct(image, n=256, norm="ortho")
            #image = tensorflow.transpose(image, [1, 0])
            image = image / np.mean(image) # Just for me.
            plt.imshow(image)
            plt.colorbar()
            # plt.title(str(np.mean((image-np.mean(image))**2)))
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')

        img = self.model.predict(img)
        filename = self.save_path + "/net_%d.png" % (epoch+1)
        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            image = img[i, :, :, 0]
            #image = tensorflow.signal.idct(image, n=256)
            #image = tensorflow.transpose(image, [1, 0])
            #image = tensorflow.signal.idct(image, n=256)
            #image = tensorflow.transpose(image, [1, 0])
            plt.imshow(image)
            plt.colorbar()
            # plt.title(str(np.mean((image-np.mean(image))**2)))
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')


######################################################################
# 2D DCT

from scipy.fftpack import dct, idct

def dct2(y, k):
    M = y.shape[0]
    N = y.shape[1]
    a = empty([M,M],float)
    b = empty([M,M],float)

    for i in range(M):
        a[i,:] = dct(y[i,:])
    for j in range(k):
        b[:,j] = dct(a[:,j])

    return b[:k, :k]


######################################################################
# 2D inverse DCT

def idct2(b, k):
    M = b.shape[0]
    N = b.shape[1]
    a = empty([M,k],float)
    y = empty([k,k],float)

    for i in range(M):
        a[i,:] = idct(b[i,:], n=k)
    for j in range(k):
        y[:,j] = idct(a[:,j], n=k)

    return y

class ZeroOneRegularizer():
    def __init__(self, alpha):
        self.alpha = alpha

    #@tf.keras.utils.register_keras_serializable(package='Custom', name='01')
    def zero_one_reg(self, weight_matrix):
        return self.alpha * tf.math.reduce_sum(weight_matrix * (1 - weight_matrix))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
