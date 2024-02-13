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
import os

import numpy as np
import pandas as pd

from .data_utils import extract_data
from .model_utils import power_iteration


def save_main_results(history, test_result, args, elapsed_training_time, current_directory):
    """
    Saves the main training and testing results to CSV files.
    
    Parameters:
        history: Training history object from Keras model training.
        test_result: Results from model evaluation on test data.
        args: Namespace object from argparse containing script arguments.
        elapsed_training_time: Float, total training time elapsed.
        current_directory: String, the path of the current working directory.
    """
    logger = logging.getLogger(__name__)
    
    # Prepare main results data
    data = {
        'data': [args.data],
        'model': [args.model],
        'seq_len': [args.seq_len],
        'pred_len': [args.pred_len],
        'lr': [args.learning_rate],
        'mse': [test_result[0]],
        'mae': [test_result[1]],
        'val_mse': [min(history.history['val_loss'])],
        'val_mae': [history.history['val_mae'][np.argmin(history.history['val_loss'])]],
        'train_mse': [history.history['loss'][np.argmin(history.history['val_loss'])]],
        'train_mae': [history.history['mae'][np.argmin(history.history['val_loss'])]],
        'training_time': elapsed_training_time,
        'rho': args.rho if args.use_sam and (args.model not in ['transformer_random']) else 0.0
    }
    df = pd.DataFrame(data)
    
    # Ensure the results directory exists
    results_dir = os.path.join(current_directory, 'results')
    os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    result_path = os.path.join(results_dir, f"result_{args.model}_{args.data}{'_sam' if args.use_sam else ''}.csv")
    
    # Save to CSV, appending if file exists, else create a new file
    df.to_csv(result_path, mode='a' if os.path.exists(result_path) else 'w', index=False, header=not os.path.exists(result_path))
    logger.info(f"Main results saved to {result_path}")


def save_training_history(history, args, current_directory):
    """
    Saves the training history, including epoch numbers, model name, prediction horizon,
    dataset used, whether SAM (Sharpness-Aware Minimization) was used, training loss, and validation loss,
    to a CSV file. The file is named based on the model, dataset, and whether SAM was employed,
    ensuring unique filenames for different training configurations.

    The function appends the new training history to the existing file if it already exists,
    allowing for cumulative recording of training sessions without overwriting previous data.

    Parameters:
        history (tf.keras.callbacks.History): The training history object returned by the fit method
                                              of a Keras model. It contains loss metrics recorded over
                                              each epoch of training.
        args (argparse.Namespace): A namespace object containing command line arguments. Expected
                                   to include 'model' (the model name), 'pred_len' (prediction horizon),
                                   'data' (the dataset name), and 'use_sam' (a boolean indicating if SAM
                                   was used during training).
        current_directory (str): The directory path where the training history CSV file will be saved.
                                 The function constructs a subdirectory named 'results' within this path
                                 to store the file.

    Outputs:
        A CSV file named 'history_[model name]_[dataset]{_sam if SAM was used}.csv' in the
        '[current_directory]/results' directory. The file contains columns for epoch number, model name,
        prediction horizon, dataset, SAM usage, training loss, and validation loss. If the file already exists,
        the function appends the new data to it, preserving any existing data.

    Example Filename:
        'results/history_modelname_datasetname_sam.csv' if SAM was used,
        'results/history_modelname_datasetname.csv' if SAM was not used.
    """
    logger = logging.getLogger(__name__)
    epochs = range(1, len(history.history['loss']) + 1)
    df = pd.DataFrame({
        'Epoch': epochs,
        'Model Name': [args.model] * len(epochs),
        'Horizon': args.pred_len,
        'Dataset': args.data,
        'Use_sam': args.use_sam,
        'Training Loss': history.history['loss'],
        'Validation Loss': history.history['val_loss']
    })
    history_csv_path = f"{current_directory}/results/history_{args.model}_{args.data}{'_sam' if args.use_sam else ''}.csv"
    df.to_csv(history_csv_path, mode='a' if os.path.exists(history_csv_path) else 'w', index=False, header=not os.path.exists(history_csv_path))
    logger.info(f"Training history saved to {history_csv_path}")

def save_additional_metrics(model, args, train_data, current_directory, capture_weights_callback):
    """
    Saves additional metrics including attention weights and sharpness, if the --add_results
    command line argument is specified. Attention weights are calculated on a batch of 32 sequences
    every 5 epochs by default, which is the capture frequency set in the CaptureWeightsCallback.
    The sharpness, measured as the largest eigenvalue of the Hessian matrix, is computed using
    power iteration at the end of the training process.

    Parameters:
        model (tf.keras.Model): The trained model.
        args (argparse.Namespace): Command line arguments specified by the user. Must include
                                   'add_results' to indicate if additional results should be saved.
        train_data (tf.data.Dataset): The training dataset used to compute sharpness.
        current_directory (str): The current working directory where results are saved.
        capture_weights_callback (CaptureWeightsCallback): The callback instance used during training
                                                           to capture attention weights.

    Note:
        - The attention weights are saved as a NumPy array (.npy) file.
        - The sharpness (largest eigenvalue of the Hessian matrix) is saved in a CSV file.
    """
    logger = logging.getLogger(__name__)
    # Check if additional results saving is requested
    if not args.add_results:
        return  # Do nothing if add_results is not True

    # Example of saving attention weights
    if hasattr(model, 'all_attention_weights'):
        attention_weights = capture_weights_callback.get_attention_weights_history()
        attention_weights_path = os.path.join(current_directory, f"results/attention_weights_{args.model}_{args.data}.npy")
        np.save(attention_weights_path, attention_weights)
        logger.info(f"Attention weights saved at {attention_weights_path}")

    # Calculate and save the eigenvalues of the Hessian matrix (sharpness)
    X_input, X_target = extract_data(train_data)
    largest_eigenvalue, delta = power_iteration(model, X_input, X_target)
    eigenvalues_data = {
        'Largest Eigenvalue': [largest_eigenvalue],
        'Delta': [delta]
    }
    eigenvalues_df = pd.DataFrame(eigenvalues_data)
    eigenvalues_path = os.path.join(current_directory, f"results/eigenvalues_{args.model}_{args.data}.csv")
    eigenvalues_df.to_csv(eigenvalues_path, index=False)
    logger.info(f"Eigenvalues (sharpness) recorded at {eigenvalues_path}")
