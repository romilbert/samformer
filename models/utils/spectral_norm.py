# coding=utf-8
# MIT License
# 
# Copyright (c) 2024 Romain Ilbert
# 
# This TensorFlow implementation is inspired by the methodology introduced in the paper
# "Stabilizing Transformer Training by Preventing Attention Entropy Collapse", and is adapted from
# the original implementation available at: https://github.com/apple/ml-sigma-reparam. This work aims
# to enhance the stability and performance of transformer models in TensorFlow by addressing the issue
# of attention entropy collapse through the sigma-reparameterization technique.
# 
# The adaptation and modifications to the original concept for TensorFlow were developed to meet
# the specific requirements of our projects, ensuring compatibility and performance within the TensorFlow
# ecosystem. This implementation facilitates the practical application of the sigma-reparameterization
# method to improve transformer training stability.
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


class SpectralNormalizedAttention(layers.MultiHeadAttention):
    """
    Spectral Normalized Multi-Head Attention Layer.
    
    This layer extends the MultiHeadAttention layer with spectral normalization on the
    query, key, and value weights, implementing the sigma-reparam method described in
    "Stabilizing Transformer Training by Preventing Attention Entropy Collapse".
    
    Paper URL: https://openreview.net/forum?id=LL8gz8FHxH
    GitHub Code: https://github.com/apple/ml-sigma-reparam
    
    Attributes:
        gamma (tf.Variable): Scaling factor for the normalized weights, trainable.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the SpectralNormalizedAttention layer with standard arguments for MultiHeadAttention."""
        super(SpectralNormalizedAttention, self).__init__(*args, **kwargs)
        self.gamma = self.add_weight(name='gamma', shape=[], initializer='ones', trainable=True)

    def build(self, input_shape):
        """Builds the layer, initializing weights."""
        super(SpectralNormalizedAttention, self).build(input_shape)
        # Additional initializations can be added here if necessary.

    def _normalize_weights(self, W):
        """
        Normalizes the weights matrix W using its spectral norm.
        
        Parameters:
            W (tf.Tensor): The weight matrix to normalize.
        
        Returns:
            tf.Tensor: Spectrally normalized weights.
        """
        singular_values = tf.linalg.svd(W, compute_uv=False)
        spectral_norm = tf.reduce_max(singular_values)
        return W / spectral_norm

    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False):
        """
        Calls the SpectralNormalizedAttention layer. Normalizes the query, key, and value weights
        before calling the parent MultiHeadAttention layer.
        
        Parameters:
            query (tf.Tensor): Query tensor.
            value (tf.Tensor): Value tensor.
            key (tf.Tensor): Key tensor. Defaults to None, in which case the query is used as the key.
            attention_mask (tf.Tensor): Optional tensor to mask out certain positions from attending to others.
            return_attention_scores (bool): Flag to return attention scores along with output.
        
        Returns:
            A tuple of (output tensor, attention scores) if return_attention_scores is True, otherwise just the output tensor.
        """
        key = query if key is None else key
        query = self._normalize_weights(query) * self.gamma
        key = self._normalize_weights(key) * self.gamma
        value = self._normalize_weights(value) * self.gamma

        return super(SpectralNormalizedAttention, self).call(query, value, key, attention_mask, return_attention_scores)
