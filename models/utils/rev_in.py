# coding=utf-8
# MIT License
# 
# Copyright (c) 2024 Romain Ilbert
# 
# This TensorFlow implementation is adapted from the official PyTorch implementation found at:
# https://github.com/ts-kim/RevIN, which is the official implementation of the paper
# "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift",
# available at: https://openreview.net/forum?id=cGDAkQo1C0p. The concept and methodology introduced
# in this paper are the foundation of this TensorFlow adaptation, aimed at enhancing time-series
# forecasting robustness and accuracy under distribution shifts.
# 
# The initial adaptation also utilized insights from Google Research code, specifically available at:
# https://github.com/google-research/google-research/blob/master/tsmixer/tsmixer_basic/models/rev_in.py,
# which has been significantly modified for broader applicability and seamless integration into various projects.
# These modifications and enhancements fit various application needs and are part of the contributions made by Romain Ilbert.
# 
# The original Google Research code is licensed under the Apache License, Version 2.0. Modifications and
# additional contributions made to adapt and enhance the functionality for TensorFlow by Romain Ilbert in 2024
# are licensed under the MIT License, as detailed above.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



"""Implementation of Reversible Instance Normalization."""

import tensorflow as tf
from tensorflow.keras import layers


class RevNorm(layers.Layer):
  """
  Implements Reversible Instance Normalization (RevNorm).

  This layer normalizes input features per instance and can reverse the normalization process. It is designed
  to maintain the statistical properties of the input data, making it particularly useful in generative models
  where the exact inverse operation is necessary.

  Attributes:
      axis (int): The axis along which to compute the mean and standard deviation for normalization.
      eps (float): A small constant added to the standard deviation to prevent division by zero.
      affine (bool): Whether to apply a learnable affine transformation after normalization.

  The original implementation can be found in the Google Research repository:
  https://github.com/google-research/google-research/blob/master/tsmixer/tsmixer_basic/models/rev_in.py

  Example usage:
      rev_norm = RevNorm(axis=-1, eps=1e-5, affine=True)
      normalized_output = rev_norm(input_tensor, mode='norm')
      denormalized_output = rev_norm(normalized_output, mode='denorm', target_slice=slice_indices)
  """

  def __init__(self, axis, eps=1e-5, affine=True):
    super().__init__()
    self.axis = axis  # Defines the dimension along which normalization is performed.
    self.eps = eps  # Small epsilon value to ensure numerical stability.
    self.affine = affine  # Determines if learnable affine parameters should be used.

  def build(self, input_shape):
    """
    Initializes the layer's weights.

    This method creates affine transformation weights if the `affine` attribute is set to True.
    It defines two trainable weights, `affine_weight` and `affine_bias`, which are used to scale and shift
    the normalized data respectively.

    Args:
        input_shape (TensorShape): The shape of the input tensor to the layer. The last dimension is used
                                   to determine the shape of the affine weights.

    Note: This method is automatically called during the first use of the layer.
    """
    if self.affine:
      self.affine_weight = self.add_weight(
          'affine_weight', shape=input_shape[-1], initializer='ones'
      )
      self.affine_bias = self.add_weight(
          'affine_bias', shape=input_shape[-1], initializer='zeros'
      )

  def call(self, x, mode, target_slice=None):
    """
    Performs normalization or denormalization on the input tensor.

    Args:
        x (Tensor): Input tensor to be normalized or denormalized.
        mode (str): 'norm' for normalization and 'denorm' for denormalization.
        target_slice (slice, optional): Target slice for denormalization.

    Returns:
        Tensor: The normalized or denormalized output.
    """
    if mode == 'norm':
      self._get_statistics(x)
      x = self._normalize(x)
    elif mode == 'denorm':
      x = self._denormalize(x, target_slice)
    else:
      raise NotImplementedError
    return x

  def _get_statistics(self, x):
    """
    Computes the mean and standard deviation of the input tensor along the specified axis.

    The calculated mean and standard deviation are used for normalizing the input data. They are computed
    using `tf.reduce_mean` and `tf.sqrt(tf.reduce_variance(...) + self.eps)` to ensure numerical stability.

    Args:
        x (Tensor): Input tensor from which the statistics are computed.

    Updates:
        self.mean (Tensor): The mean of the input tensor, calculated along the specified axis.
        self.stdev (Tensor): The standard deviation of the input tensor, ensuring numerical stability by adding `self.eps`.
    """
    self.mean = tf.stop_gradient(
        tf.reduce_mean(x, axis=self.axis, keepdims=True)
    )
    self.stdev = tf.stop_gradient(
        tf.sqrt(
            tf.math.reduce_variance(x, axis=self.axis, keepdims=True) + self.eps
        )
    )

  def _normalize(self, x):
    """
    Normalizes the input tensor using the computed mean and standard deviation.

    This method subtracts the mean from the input tensor and divides it by the standard deviation, effectively
    standardizing the input to have a mean of 0 and a standard deviation of 1. If affine transformation is enabled,
    it further applies scaling and shifting to the standardized input.

    Args:
        x (Tensor): Input tensor to be normalized.

    Returns:
        Tensor: The normalized tensor.
    """
    x = x - self.mean
    x = x / self.stdev
    if self.affine:
      x = x * self.affine_weight
      x = x + self.affine_bias
    return x

  def _denormalize(self, x, target_slice=None):
    """
    Reverses the normalization process for the given slice of the input tensor.

    This method applies the inverse of the normalization operation. If affine transformation was applied during
    normalization, it reverses this process first. Then, it multiplies the tensor by the standard deviation and adds
    the mean to denormalize the data.

    Args:
        x (Tensor): Normalized tensor that needs to be denormalized.
        target_slice (slice, optional): Specific slice of the tensor to denormalize. Useful when different parts
                                         of the tensor require different reverse operations.

    Returns:
        Tensor: The denormalized tensor.
    """
    if self.affine:
      x = x - self.affine_bias[target_slice]
      x = x / self.affine_weight[target_slice]
    x = x * self.stdev[:, :, target_slice]
    x = x + self.mean[:, :, target_slice]
    return x
