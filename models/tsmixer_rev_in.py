# Adapted from "TSMixer: An all-MLP Architecture for Time Series Forecasting" DataLoader implementation
# Original authors: Si-An Chen, Chun-Liang Li, Nate Yoder, Sercan Arik, and Tomas Pfister
# Project link: https://github.com/google-research/google-research/tree/master/tsmixer
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications made by Romain Ilbert in 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications made under the MIT License by Romain Ilbert in 2024.
# The following is added under the MIT License, fulfilling the conditions of the Apache License, Version 2.0:
#
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

from tensorflow.keras import layers, Model
import tensorflow as tf
from .utils import RevNorm, SAM

class TSMixerModel(Model):
    """
    Implementation of TSMixer with Reversible Instance Normalization.
    
    This model incorporates reversible instance normalization, offering an option for
    Sharpness-Aware Minimization (SAM), and supports customizable architecture
    through various parameters.

    For more details on TSMixer, see the original paper:
    - [TSMixer: An All-MLP Architecture for Time Series Forecasting](https://openreview.net/forum?id=wbpxTuXgm0)

    The original implementation can be found at:
    - https://github.com/google-research/google-research/tree/master/tsmixer
    
    Attributes:
        model_input_shape (Tuple[int, int]): Shape of the model input.
        pred_len (int): Prediction length.
        use_sam (bool): Whether to use Sharpness-Aware Minimization.
        norm_type (str): Type of normalization ('L' for LayerNorm, otherwise BatchNorm).
        activation (str): Activation function for the dense layers.
        dropout (float): Dropout rate.
        ff_dim (int): Dimension of the feed-forward network.
        n_blocks (int): Number of blocks in the model.
        rho (float): Hyperparameter for SAM, if used.
    """

    def __init__(self, model_input_shape, pred_len, use_sam=None, norm_type=None, 
                 activation="relu", dropout=0.1, ff_dim=128, n_blocks=1, rho=0.0):
        super(TSMixerModel, self).__init__()
        self.model_input_shape = model_input_shape
        self.pred_len = pred_len
        self.use_sam = use_sam
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = dropout
        self.ff_dim = ff_dim
        self.n_blocks = n_blocks
        self.rho = rho

        self.rev_norm = RevNorm(axis=-2)
        self.temporal_dense = layers.Dense(model_input_shape[-2], activation=activation, name="temporal_dense")
        self.feature_dense = layers.Dense(ff_dim, activation=activation, name="feature_dense")
        self.output_dense = layers.Dense(model_input_shape[-1], name="output_dense")
        self.layer_norm = layers.LayerNormalization(axis=[-2, -1])
        self.batch_norm = layers.BatchNormalization(axis=[-2, -1])
        self.dense = layers.Dense(pred_len)

    def call(self, inputs):
        """
        Forward pass of the TSMixerModel.
        
        Parameters:
            inputs (tf.Tensor): Input tensor.
            training (bool): Indicates whether the forward pass is for training.
        
        Returns:
            tf.Tensor: The model's output tensor.
        """
        x = self.rev_norm(inputs, mode='norm')
        for _ in range(self.n_blocks):
            x = self.res_block(x, self.norm_type, self.activation, self.dropout, self.ff_dim)

        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.dense(x)
        outputs = tf.transpose(x, perm=[0, 2, 1])
        outputs = self.rev_norm(outputs, mode='denorm')
        return outputs
    
    def res_block(self, inputs, norm_type, dropout):
        """
        Defines a residual block for the TSMixer model.
        
        Parameters:
            inputs (tf.Tensor): Input tensor to the residual block.
            norm_type (str): Type of normalization to apply.
            activation (str): Activation function.
            dropout (float): Dropout rate.
            ff_dim (int): Dimensionality of the feature dense layer.
        
        Returns:
            tf.Tensor: Output tensor of the residual block.
        """
        norm_layer = self.layer_norm if norm_type == 'L' else self.batch_norm

        # Temporal Linear
        x = norm_layer(inputs)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.temporal_dense(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feature Linear
        x = norm_layer(res)
        x = self.feature_dense(x)
        x = layers.Dropout(dropout)(x)
        x = self.output_dense(x)
        x = layers.Dropout(dropout)(x)

        return x + res

    def train_step(self, data):
        """
        Custom training logic, including the first and second steps of SAM optimization.
        
        Parameters:
            data (tuple): A tuple of input data and labels.
        
        Returns:
            dict: A dictionary mapping metric names to their current value.
        """
        x, y = data

        if self.use_sam:
            sam_optimizer = SAM(self.optimizer, rho=self.rho, eps=1e-12)
        else:
            # Fallback to the default optimizer if SAM is not used.
            sam_optimizer = self.optimizer

        # SAM's first step
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        sam_optimizer.first_step(gradients, self.trainable_variables)

        # SAM's second step
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        sam_optimizer.second_step(gradients, self.trainable_variables)

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
