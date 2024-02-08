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
import math
import sys

import tensorflow as tf
from tqdm import tqdm

from models import BaseModel, TSMixerModel


def power_iteration(model, inputs, targets, num_iterations=50):
    """
    Computes the dominant eigenvalue of the model's Hessian matrix.
    
    Parameters:
        model (tf.keras.Model): The model to evaluate.
        inputs (np.ndarray): Model inputs.
        targets (np.ndarray): Model targets.
        num_iterations (int): Number of iterations for power iteration.
        
    Returns:
        Tuple[float, float]: The dominant eigenvalue and the delta of the last two iterations.
    """
    logger = logging.getLogger(__name__)
    tf.random.set_seed(42)
    v = [tf.random.normal(shape=w.shape) for w in model.trainable_variables]
    last_rayleigh_quotient = 0
    for iteration in tqdm(range(num_iterations)):
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                predictions = model(inputs)
                loss = tf.keras.losses.mean_squared_error(predictions, targets)
            grads = tape1.gradient(loss, model.trainable_variables)
        hessian_vector = tape2.gradient(grads, model.trainable_variables, output_gradients=v)

        v_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(hv)) for hv in hessian_vector]))
        v = [hv / v_norm for hv in hessian_vector]

        rayleigh_quotient = sum([tf.reduce_sum(g * hv) for g, hv in zip(grads, hessian_vector)])
        rayleigh_quotient = abs(rayleigh_quotient.numpy())

        if iteration == num_iterations - 2:
            last_rayleigh_quotient = rayleigh_quotient

    delta = abs(rayleigh_quotient - last_rayleigh_quotient)
    logger.info(f"Delta: {delta}")
    return rayleigh_quotient, delta

def cosine_annealing(epoch, max_epochs, initial_lr, min_lr):
    """
    Applies cosine annealing to the learning rate.
    
    Parameters:
        epoch (int): Current epoch.
        max_epochs (int): Maximum number of epochs.
        initial_lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.
        
    Returns:
        float: Adjusted learning rate.
    """
    cos_inner = (math.pi * (epoch % max_epochs)) / max_epochs
    return min_lr + (initial_lr - min_lr) * (math.cos(cos_inner) + 1) / 2

def create_optimizer(args):
    """
    Initialize the optimizer based on provided command line arguments.

    Parameters:
    - args: Parsed command line arguments.

    Returns:
    - A TensorFlow optimizer instance.
    """
    return tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

def initialize_model(args, n_features):
    """
    Initializes and configures the model specified by the command-line arguments for time series forecasting tasks.
    This function supports a dynamic selection among various models, including TSMixer, a linear BaseModel as a placeholder,
    and Transformer-based models, accommodating a wide range of architectural strategies tailored to time series data.

    Parameters:
        args (argparse.Namespace): Contains all the command-line arguments provided by the user, detailing the model
                                   configuration, data handling specifications, and training hyperparameters.
        n_features (int): The number of features in the input data, critical for defining the model's input layer size.

    Returns:
        tf.keras.Model: A TensorFlow Keras model instance, fully initialized and ready for training, based on the 
                        specifications provided in `args`.
    """
    model_kwargs = {
        'model_input_shape': (args.seq_len, n_features),  # Define the shape of the input data.
        'pred_len': args.pred_len,  # Specify the prediction length for the model output.
        'norm_type': args.norm_type,  # Choose the type of normalization (Layer or Batch normalization).
        'activation': args.activation,  # Set the activation function for neural network layers.
        'dropout': args.dropout,  # Determine the dropout rate for regularization.
        'rho': args.rho  # Specify the rho parameter for Sharpness-Aware Minimization (SAM), if applicable.
    }

    # Conditionally disable reversible instance normalization (RevIn) based on dataset type or additional results logging.
    use_revin = not (args.data == 'toy' or args.add_results)
    
    if args.model == 'tsmixer':
        # Initialize TSMixer model
        model = TSMixerModel(**model_kwargs, n_blocks=args.n_block, ff_dim=args.ff_dim, use_sam=args.use_sam)
    elif args.model == 'linear':
        # Initialize a baseline linear model, projecting inputs directly to outputs without attention mechanisms.
        model = BaseModel(**model_kwargs, use_attention=False, use_sam=args.use_sam)
    elif args.model in ['transformer', 'transformer_random', 'spectrans']:
        model_kwargs.update({
            'num_heads': args.num_heads,  # Define the number of attention heads.
            'd_model': args.d_model,  # Set the dimensionality of the model's embeddings.
            'use_attention': True,  # Enable the use of attention mechanisms.
            'use_revin': use_revin and args.model != 'transformer_random',  # Conditionally apply RevIn.
            'trainable': args.model not in ['transformer_random'],  # Specify if the model is trainable.
            'spec': args.model == 'spectrans',  # Indicate the use of sigma reparam, if selected.
            'use_sam': args.use_sam  # Apply Sharpness-Aware Minimization, if enabled.
        })
        model = BaseModel(**model_kwargs)
    else:
        raise ValueError(f"Model '{args.model}' is not supported.")

    return model

def compile_model(model, optimizer):
    """
    Compiles the TensorFlow model with necessary configurations.

    Parameters:
    - model: The TensorFlow model to compile.
    - optimizer: The optimizer to use for compiling the model.

    This function compiles the model with Mean Squared Error loss and Mean Absolute Error metric,
    and configures it to run eagerly for more intuitive debugging.
    """
    try:
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'], run_eagerly=True)
    except Exception as e:
        logging.error(f"Error during model compilation: {e}")
        sys.exit

def log_model_info(model, args):
    """
    Logs information about the model and training configuration.

    Parameters:
    - model: The TensorFlow model.
    - args: Parsed command line arguments.
    """
    logging.info(f"Initialized model: {args.model}")
    logging.info(f"Dataset: {args.data}")
    logging.info(f"Prediction horizon (pred_len): {args.pred_len}")

    # Enhanced attribute checking and logging
    if hasattr(model, 'use_revin'):
        logging.info(f"Reversible instance normalization (RevIn): {'Enabled' if model.use_revin else 'Disabled'}")
    if hasattr(model, 'spec'):
        logging.info(f"Spectral reparametrization (spec): {'Enabled' if model.spec else 'Disabled'}")
    if hasattr(model, 'trainable'):
        logging.info(f"Attention trainability (trainable): {'Enabled' if model.trainable else 'Disabled'}")

    logging.info(f"{'Using SAM with rho=' + str(args.rho) if args.use_sam else 'Not using SAM'}")