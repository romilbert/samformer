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

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

from .model_utils import cosine_annealing


class CaptureWeightsCallback(tf.keras.callbacks.Callback):
    """
    A custom callback to capture and analyze the weights of a model during training.

    Attributes:
    - model (tf.keras.Model): The model from which weights are captured.
    - penultimate_weights (list): Stored attention weights.
    - attention_weights_history (list): History of attention weights captured during training.
    """
    
    def __init__(self, model):
        """
        Initialize the callback with a specific model.
        """
        super().__init__()
        self.model = model
        self.penultimate_weights = None
        self.attention_weights_history = []

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch during training to capture and analyze weights.

        Parameters:
        - epoch (int): The current epoch number.
        - logs (dict): Currently unused parameter. Contains logs from the training epoch.
        """
        if epoch % 5 == 0:  # Perform analysis every 5 epochs
            # Retrieve attention weights from the model
            last_attention_weights = self.model.get_last_attention_weights()
            if last_attention_weights is not None:
                self.attention_weights_history.append(last_attention_weights)
    
    def get_attention_weights_history(self):
        """Return the history of attention weights captured."""
        return self.attention_weights_history




def setup_callbacks(args, checkpoint_path, model):
    """
    Configures training callbacks.

    Args:
        args: Command line arguments.
        checkpoint_path: Path for saving the best model.

    Returns:
        List of configured TensorFlow callbacks.
    """
    try:
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',  
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            verbose=1,
        )

        lr_schedule_callback = LearningRateScheduler(
            lambda epoch: cosine_annealing(epoch, args.train_epochs, args.learning_rate, 1e-6),
            verbose=1,
        )

        # Assuming CaptureWeightsCallback is implemented correctly
        capture_weights_callback = CaptureWeightsCallback(model)

        callbacks = [checkpoint_callback, lr_schedule_callback, capture_weights_callback, early_stop_callback]
        return callbacks, capture_weights_callback
    except Exception as e:
        logging.error(f"Error setting up callbacks: {e}")
        raise