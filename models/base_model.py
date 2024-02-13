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

"""Implementation of our Transformer with Reversible Instance Normalization and Channel-Wise Attention."""

import tensorflow as tf
from tensorflow.keras import layers
import collections
from models.utils import RevNorm, SAM, SpectralNormalizedAttention

class BaseModel(tf.keras.Model):
    """
    A base model class that integrates various enhancements including 
    Reversible Instance Normalization and Channel-Wise Attention, and optionally,
    spectral normalization and SAM optimization. To use SAMformer, enable use_sam,
    use_attention, use_revin and trainable.
    
    Attributes:
        pred_len (int): The length of the output predictions.
        num_heads (int): The number of heads in the multi-head attention mechanism. 
        d_model (int): The dimensionality of the embedding vectors.
        use_sam (bool): If True, applies Sharpness-Aware Minimization (SAM) optimization technique during training, 
                        aiming to improve model generalization by considering the loss landscape's sharpness.
        use_attention (bool): If True, enables the multi-head attention mechanism in the model. If False, the model is
                              equivalent to a simple linear layer.
        use_revin (bool): If True, applies Reversible Instance Normalization (RevIN) to the model.
        trainable (bool): Specifies if the model's weights should be updated or frozen during training. Useful to 
                          highlight some attention layer issues in Time Series Forecasting.
        rho (float): The neighborhood size parameter for SAM optimization. It determines the radius within which SAM 
                     seeks to minimize the sharpness of the loss landscape.
        spec (bool): If True, applies spectral normalization (sigma-reparam) to the attention mechanism, aiming to
                     stabilize the training by constraining the spectral norm of the weight matrices.
        
    Methods:
        call(inputs, training=False): Defines the computation from inputs to outputs, optionally applying SAM, 
                                      spectral normalization, and reversible instance normalization based on the 
                                      configuration.
        _apply_attention(x): Applies the attention mechanism to the input tensor, capturing the inter-dependencies 
                             within the data thanks to the Channel-Wise Attention mechanism.
        get_last_attention_weights(): Retrieves the attention weights from the last but one batch, useful for 
                                      analysis and debugging purposes.
        train_step(data): Custom training logic, including the application of SAM's two-step optimization process, 
                          to improve model generalization and performance stability.

    """

    def __init__(self, pred_len, num_heads=1, d_model=16, use_sam=None, 
                 use_attention=None, use_revin=None, 
                 trainable=None, rho=None, spec=None):
        super(BaseModel, self).__init__()
        self.pred_len = pred_len
        self.num_heads = num_heads
        self.d_model = d_model
        self.use_sam = use_sam
        self.use_attention = use_attention
        self.use_revin = use_revin
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

        #Define trainability of attention layer
        self.attention_layer.trainable = trainable

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

    def train_step(self, data):
        sam_optimizer = SAM(self.optimizer, rho=self.rho, eps=1e-12) 

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
