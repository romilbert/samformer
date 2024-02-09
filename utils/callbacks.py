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
    A custom callback designed to capture and analyze the weights of a model during training, specifically
    focusing on capturing attention weights at specified intervals.

    Attributes:
        model (tf.keras.Model): The model from which weights are captured.
        attention_weights_history (list): A history of attention weights captured during training, stored
                                          at intervals specified by the training routine (default is every 5 epochs).

    Methods:
        on_epoch_end(epoch, logs=None): Captures attention weights from the model at the end of specified epochs.
        get_attention_weights_history(): Returns the collected history of attention weights.
    """
    
    def __init__(self, model):
        """
        Initializes the callback with a specific model to monitor its attention weights during training.

        Parameters:
            model (tf.keras.Model): The model whose attention weights are to be monitored and captured.
        """
        super().__init__()
        self.model = model
        self.penultimate_weights = None
        self.attention_weights_history = []

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch during training to capture and store attention weights if the current
        epoch satisfies the capture criteria (e.g., every 5 epochs).

        Parameters:
            epoch (int): The current epoch number.
            logs (dict): Currently unused. Contains logs from the training epoch.
        """
        if epoch % 5 == 0:  # Perform analysis every 5 epochs
            # Retrieve attention weights from the model
            last_attention_weights = self.model.get_last_attention_weights()
            if last_attention_weights is not None:
                self.attention_weights_history.append(last_attention_weights)
    
    def get_attention_weights_history(self):
        """
        Returns the history of attention weights captured during training.

        Returns:
            A list of attention weights captured at specified intervals during training.
        """
        return self.attention_weights_history




def setup_callbacks(args, checkpoint_path, model):
    """
    Configures and returns a list of TensorFlow callbacks for training, including model checkpointing,
    early stopping, learning rate scheduling, and custom weight capture functionality. This function
    also handles the creation of a `CaptureWeightsCallback` for capturing model weights during training,
    which is returned alongside the list of callbacks for further use.

    Args:
        args (argparse.Namespace): Command line arguments provided to the training script. Expected
                                   to contain 'patience' (for early stopping), 'train_epochs' (total number
                                   of training epochs), and 'learning_rate' (initial learning rate).
        checkpoint_path (str): Path to the directory where the model checkpoints will be saved. The best
                               performing model according to validation loss will be saved to this location.
        model (tf.keras.Model): The Keras model being trained. Required for some callbacks, such as
                                the custom `CaptureWeightsCallback` which captures model weights.

    Returns:
        tuple: A tuple containing two elements:
            - List of configured TensorFlow callbacks, including `ModelCheckpoint`, `EarlyStopping`,
              `LearningRateScheduler`, and `CaptureWeightsCallback`.
            - The `CaptureWeightsCallback` instance, which can be used to access captured weights after training.

    Raises:
        Exception: If an error occurs during the setup of callbacks, an exception is logged and raised.

    Example Usage:
        >>> callbacks, capture_weights_callback = setup_callbacks(args, './checkpoints', model)
        This prepares the training environment with necessary callbacks and allows for weight capture analysis.
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

        capture_weights_callback = CaptureWeightsCallback(model)

        callbacks = [checkpoint_callback, lr_schedule_callback, capture_weights_callback, early_stop_callback]
        return callbacks, capture_weights_callback
    except Exception as e:
        logging.error(f"Error setting up callbacks: {e}")
        raise