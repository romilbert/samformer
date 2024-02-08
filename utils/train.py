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
import tensorflow as tf


def train_model(model, train_data, val_data, args, callbacks):
    """
    Runs the model training process.

    Args:
        model: Compiled TensorFlow model ready for training.
        train_data: Training dataset.
        val_data: Validation dataset.
        args: Command line arguments.
        callbacks: List of configured TensorFlow callbacks.

    Returns:
        The history object returned from model.fit().
    """
    logger = logging.getLogger(__name__)
    try:
        history = model.fit(
            train_data,
            epochs=args.train_epochs,
            validation_data=val_data,
            callbacks=callbacks,
        )
        return history
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
