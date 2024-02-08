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

"""Implementation of SAMformer with Reversible Instance Normalization and Channel-Wise Attention."""

import tensorflow as tf
from tensorflow.keras import layers
import collections
from rev_in import RevNorm
from spectral_norm import SpectralNormalizedAttention
from sam import SAM

class BaseModel(tf.keras.Model):
    """
    A base model class that integrates various enhancements including 
    reversible instance normalization, multi-head attention, and optionally,
    spectral normalization and SAM optimization.
    
    Attributes:
        input_shape (Tuple[int, int]): The shape of the input data.
        pred_len (int): The length of the prediction.
        num_heads (int): The number of attention heads.
        d_model (int): The dimensionality of the model.
        use_sam (bool): Flag to indicate the usage of SAM optimization.
        use_attention (bool): Flag to indicate the usage of attention mechanism.
        norm_type (str): Type of normalization to be applied.
        activation (str): Activation function to be used.
        dropout (float): Dropout rate.
        ff_dim (int): Dimension of the feed-forward network.
        n_blocks (int): Number of blocks.
        use_blocks (bool): Flag to indicate the usage of blocks.
        use_revin (bool): Flag to indicate the usage of reversible instance normalization.
        trainable (bool): If the model should be trainable.
        rho (float): Hyperparameter for SAM, if used.
        spec (bool): Flag to indicate the usage of spectral normalization.
    """

    def __init__(self, model_input_shape, pred_len, num_heads=1, d_model=16, use_sam=None, 
                 use_attention=None, norm_type=None, activation=None, dropout=None, 
                 ff_dim=None, n_blocks=None, use_blocks=False, use_revin=None, 
                 trainable=None, rho=None, spec=None):
        super(BaseModel, self).__init__()
        #self.model_input_shape = model_input_shape
        self.pred_len = pred_len
        self.num_heads = num_heads
        self.d_model = d_model
        self.use_sam = use_sam
        self.use_attention = use_attention
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = dropout
        self.ff_dim = ff_dim
        self.n_blocks = n_blocks
        self.use_blocks = use_blocks
        self.use_revin = use_revin
        self.trainable = trainable
        self.rho = rho if use_sam and trainable else 0.0
        self.spec = spec

        # Define model layers
        self.rev_norm = RevNorm(axis=-2)
        self.attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dense = layers.Dense(pred_len)
        self.all_attention_weights = collections.deque(maxlen=2)
        self.all_dense_weights = collections.deque(maxlen=2)

        if self.spec:
            self.spec_layer = SpectralNormalizedAttention(num_heads=num_heads, key_dim=d_model)

    def call(self, inputs, training=False):
        """
        The forward pass for the model.
        
        Parameters:
            inputs (Tensor): Input tensor.
            training (bool): Whether the call is for training.
        
        Returns:
            Tensor: The output of the model.
        """
        x = inputs
        if self.use_revin:
            x = self.rev_norm(x, mode='norm')
        x = tf.transpose(x, perm=[0, 2, 1])

        if self.use_attention:
            attention_output = self._apply_attention(x)
            x = layers.Add()([x, attention_output])

        x = self.dense(x)
        outputs = tf.transpose(x, perm=[0, 2, 1])

        if self.use_revin:
            outputs = self.rev_norm(outputs, mode='denorm')

        return outputs

    def _apply_attention(self, x):
        """
        Applies the attention mechanism to the input tensor.
        
        Parameters:
            x (Tensor): The input tensor.
            training (bool): Whether the call is for training.
        
        Returns:
            Tensor: The output tensor after applying attention.
        """
        if self.spec:
            attention_output, weights = self.spec_layer(x, x, return_attention_scores=True)
        else:
            attention_output, weights = self.attention_layer(x, x, return_attention_scores=True)
        
        self.all_attention_weights.append(weights.numpy())
        return attention_output

    def get_last_attention_weights(self):
        """Returns the attention weights from the last but one batch."""
        if len(self.all_attention_weights) > 1:
            return self.all_attention_weights[-2]
        return None

    def get_last_dense_weights(self):
        """Returns the dense layer weights from the last but one batch."""
        if len(self.all_dense_weights) > 1:
            return self.all_dense_weights[-2]
        return None

    def train_step(self, data):
        # Instancier SAM ici ou le passer en tant qu'attribut du modèle
        sam_optimizer = SAM(self.optimizer, rho=self.rho, eps=1e-12) #0.9 ou 1 avec les configs bien d_model et n_heads

        # Unpack the data.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Apply SAM's first step
        sam_optimizer.first_step(gradients, self.trainable_variables)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass again
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients again
        gradients = tape.gradient(loss, self.trainable_variables)

        # Apply SAM's second step
        sam_optimizer.second_step(gradients, self.trainable_variables)

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
