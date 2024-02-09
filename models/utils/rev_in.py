# MIT License
# 
# Copyright (c) 2024 Romain Ilbert
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

import tensorflow as tf
from tensorflow.keras import layers


class RevNorm(layers.Layer):
    """
    Reversible Instance Normalization (RevIN) Layer.
    
    This layer implements reversible instance normalization, allowing for
    normalization and denormalization processes in neural networks. It supports
    optional affine transformation.
    
    For more details on the Reversible Instance Normalization, see the original paper:
    - [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://openreview.net/forum?id=cGDAkQo1C0p)

    The original implementation can be found at:
    - https://github.com/ts-kim/RevIN
    
    Attributes:
        axis (int): The axis to normalize across. Typically, this would be the features axis.
        eps (float): A small epsilon value to avoid division by zero during normalization.
        affine (bool): Whether to include an affine transformation as part of normalization.
    """

    def __init__(self, axis=-1, eps=1e-5, affine=True, **kwargs):
        super(RevNorm, self).__init__(**kwargs)
        self.axis = axis
        self.eps = eps
        self.affine = affine

    def build(self, input_shape):
        """
        Initializes the layer's weights.

        Parameters:
            input_shape (TensorShape): The shape of the input tensor.
        """
        if self.affine:
            self.affine_weight = self.add_weight(
                name='affine_weight',
                shape=(input_shape[-1],),
                initializer='ones',
                trainable=True
            )
            self.affine_bias = self.add_weight(
                name='affine_bias',
                shape=(input_shape[-1],),
                initializer='zeros',
                trainable=True
            )
        super().build(input_shape)

    def call(self, x, mode='norm', target_slice=None):
        """
        Applies normalization or denormalization to the input tensor.

        Parameters:
            x (tf.Tensor): Input tensor.
            mode (str): 'norm' for normalization and 'denorm' for denormalization.
            target_slice (slice): The specific slice of the tensor to denormalize. Only used in 'denorm' mode.

        Returns:
            tf.Tensor: The normalized or denormalized tensor.
        """
        if mode == 'norm':
            mean, variance = self._get_statistics(x)
            x = self._normalize(x, mean, variance)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else:
            raise ValueError("Unsupported mode. Use 'norm' or 'denorm'.")
        return x

    def get_config(self):
        """
        Returns the configuration of the layer.
        """
        config = super(RevNorm, self).get_config()
        config.update({
            "axis": self.axis,
            "eps": self.eps,
            "affine": self.affine
        })
        return config

    def _get_statistics(self, x):
        """
        Computes the mean and variance of the input tensor along the specified axis.

        Parameters:
            x (tf.Tensor): Input tensor.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The mean and variance tensors.
        """
        self.mean = tf.stop_gradient(
            tf.reduce_mean(x, axis=self.axis, keepdims=True)
        )
        self.stdev = tf.stop_gradient(
            tf.sqrt(
                tf.math.reduce_variance(x, axis=self.axis, keepdims=True) + self.eps
            )
        )
        return self.mean, self.stdev

    def _normalize(self, x, mean, variance):
        """
        Normalizes the input tensor using the computed mean and variance.

        Parameters:
            x (tf.Tensor): Input tensor.
            mean (tf.Tensor): Mean tensor.
            variance (tf.Tensor): Variance tensor.

        Returns:
            tf.Tensor: The normalized tensor.
        """
        x = (x - mean) / tf.sqrt(variance)
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        """
        Denormalizes the input tensor. This method reverses the normalization effect.

        Parameters:
            x (tf.Tensor): The normalized tensor.
            target_slice (slice): The specific slice of the tensor to denormalize.

        Returns:
            tf.Tensor: The denormalized tensor.
        """
        if self.affine:
            x = (x - self.affine_bias[target_slice]) / self.affine_weight[target_slice]
        return x
