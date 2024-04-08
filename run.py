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

import argparse
import logging
import os
import sys
import time
import tensorflow as tf
from utils import (
    compile_model,
    configure_environment,
    create_optimizer,
    initialize_model,
    load_data,
    log_model_info,
    save_additional_metrics,
    save_main_results,
    save_training_history,
    setup_callbacks,
    setup_experiment_id,
    train_model
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    """Parses command line arguments for the training experiment."""
    parser = argparse.ArgumentParser(description="Train models for Time Series Forecasting.")

    parser.add_argument("--model", type=str, default="tsmixer",
                        choices=["tsmixer", "transformer", "transformer_random", "spectrans", "linear"],
                        help="Model to train.")

    parser.add_argument("--use_sam", action="store_true",
                        help="Whether to use SAM (Sharpness-Aware Minimization).")

    parser.add_argument("--data", type=str, default="weather",
                        choices=["electricity", "exchange_rate", "weather", "ETTm1", "ETTm2", "ETTh1", "ETTh2", "traffic", "toy"],
                        help="Dataset for training.")

    parser.add_argument("--feature_type", type=str, default="M",
                        choices=["S", "M", "MS"],
                        help="Type of forecasting task.")

    parser.add_argument("--target", type=str, default="OT",
                        help="Target feature for S or MS task.")

    parser.add_argument("--seq_len", type=int, default=336,
                        help="Input sequence length.")

    parser.add_argument("--pred_len", type=int, default=96,
                        help="Prediction sequence length.")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training.")

    parser.add_argument("--train_epochs", type=int, default=100,
                        help="Total number of training epochs.")

    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate for optimizer.")

    parser.add_argument("--rho", type=float, default=0.7,
                        help="Rho parameter for SAM, if used.")

    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping.")

    parser.add_argument("--n_block", type=int, default=2,
                        help="Number of blocks in the model architecture.")

    parser.add_argument("--ff_dim", type=int, default=2048,
                        help="Dimension of feed-forward layers.")

    parser.add_argument("--num_heads", type=int, default=1,
                        help="Number of heads in multi-head attention layers.")

    parser.add_argument("--d_model", type=int, default=16,
                        help="Dimensionality of the model embeddings.")

    parser.add_argument("--dropout", type=float, default=0.05,
                        help="Dropout rate.")

    parser.add_argument("--norm_type", type=str, default="B", choices=["L", "B"],
                        help="Normalization type: LayerNorm (L) or BatchNorm (B).")

    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"],
                        help="Activation function.")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    parser.add_argument("--checkpoint_dir", type=str, 
                        default="checkpoints",
                        help="Directory to save model checkpoints.")

    parser.add_argument("--delete_checkpoint", action="store_true",
                        help="Whether to delete model checkpoints after training.")

    parser.add_argument("--result_path", type=str, default="results.csv",
                        help="Path to save the training results.")

    parser.add_argument("--add_results", action="store_true",
                        help="Whether to save additional results.")

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Configure the execution environment
    current_directory = configure_environment()
    
    # Establish experiment identification
    exp_id = setup_experiment_id(args)
    logging.info(f"Experiment ID: {exp_id}")
    
    # Data loading with a clear distinction for toy data
    if args.data == 'toy':
        train_data, val_data, test_data = load_data(args)
        n_features = 7
    else : 
        train_data, val_data, test_data, n_features = load_data(args)
    
    # Model initialization and configuration logging
    model = initialize_model(args, n_features)
    log_model_info(model, args)

    # Optimizer setup and model compilation
    optimizer = create_optimizer(args)
    compile_model(model, optimizer)

    # Callbacks configuration for model checkpoints
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{exp_id}_best.h5")
    callbacks, capture_weights_callback = setup_callbacks(args, checkpoint_path, model)

    # Training process initiation
    start_training_time = time.time()
    history = train_model(model, train_data, val_data, args, callbacks)
    elapsed_training_time = time.time() - start_training_time
    logging.info(f"Training completed in {elapsed_training_time:.2f} seconds.")

    # Model evaluation on test data with specific exception handling
    model.load_weights(checkpoint_path)
    try:
        test_result = model.evaluate(test_data)
    except tf.errors.OpError as e: 
        logging.error(f"Error during model evaluation: {e}")
        sys.exit(1)

    # Checkpoint cleanup with logging
    if args.delete_checkpoint and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logging.info("Checkpoint files have been deleted.")

    # Results saving and history logging
    save_main_results(history, test_result, args, elapsed_training_time, current_directory)
    save_training_history(history, args, current_directory)

    # Additional metrics storage based on user request
    if args.add_results:
        save_additional_metrics(model, args, train_data, current_directory, capture_weights_callback)

if __name__ == '__main__':
    main()

