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


import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


class TSFDataLoader:
    """
    DataLoader for time series forecasting datasets.

    Handles the loading, preprocessing, and dataset splitting for time series forecasting,
    including support for univariate and multivariate series.
    """

    def __init__(self, data, batch_size, seq_len, pred_len, feature_type, target='OT'):
        """
        Initializes the DataLoader with dataset configurations.

        Parameters:
            data (str): Identifier for the dataset.
            batch_size (int): Batch size for the data loader.
            seq_len (int): Length of input sequences.
            pred_len (int): Length of prediction sequences.
            feature_type (str): Type of features to use ('S', 'M', 'MS').
            target (str): Target variable for forecasting. Defaults to 'OT'.
        """
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target

        self.current_directory = os.getcwd()
        self.LOCAL_CACHE_DIR = os.path.join(self.current_directory, "dataset")

        self._ensure_data_directory()
        self._read_and_process_data()

    def _ensure_data_directory(self):
        """Ensures that the local cache directory for dataset exists."""
        if not os.path.exists(self.LOCAL_CACHE_DIR):
            os.makedirs(self.LOCAL_CACHE_DIR)

    def _read_and_process_data(self):
        """
        Reads the dataset from a file, preprocesses it, and splits it into
        training, validation, and test sets.
        """
        file_name = f"{self.data}.csv"
        cache_filepath = os.path.join(self.LOCAL_CACHE_DIR, file_name)

        df_raw = pd.read_csv(cache_filepath)
        df = df_raw.set_index('date')

        if self.feature_type == 'S':
            df = df[[self.target]]
        elif self.feature_type == 'MS':
            target_idx = df.columns.get_loc(self.target)
            self.target_slice = slice(target_idx, target_idx + 1)

        self._split_data(df)
        self._scale_data()

    def _split_data(self, df):
        """
        Splits the dataframe into training, validation, and test sets.
        """
        train_end = int(len(df) * 0.7)
        val_end = train_end + int(len(df) * 0.2)

        self.train_df = df[:train_end]
        self.val_df = df[train_end - self.seq_len:val_end]
        self.test_df = df[val_end - self.seq_len:]

    def _scale_data(self):
        """
        Scales the data using standard scaling based on the training set.
        """
        self.scaler = StandardScaler()
        self.train_df = pd.DataFrame(self.scaler.fit_transform(self.train_df), columns=self.train_df.columns, index=self.train_df.index)
        self.val_df = pd.DataFrame(self.scaler.transform(self.val_df), columns=self.val_df.columns, index=self.val_df.index)
        self.test_df = pd.DataFrame(self.scaler.transform(self.test_df), columns=self.test_df.columns, index=self.test_df.index)

    def _make_dataset(self, df, shuffle):
        """
        Converts a DataFrame into a tf.data.Dataset.

        Parameters:
            df (pd.DataFrame): DataFrame to convert.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            tf.data.Dataset: Resulting dataset.
        """
        data = np.array(df, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.seq_len + self.pred_len,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=self.batch_size,
        )
        return ds.map(self._split_window)

    def _split_window(self, features):
        """
        Splits the input features into input sequences and target sequences.

        Parameters:
            features (tuple): Tuple of features.

        Returns:
            tuple: Tuple of (input_sequence, target_sequence).
        """
        inputs = features[:, :self.seq_len]
        targets = features[:, self.seq_len:]
        return inputs, targets

    def get_train(self):
        """Returns the training dataset."""
        return self._make_dataset(self.train_df, shuffle=True)

    def get_val(self):
        """Returns the validation dataset."""
        return self._make_dataset(self.val_df, shuffle=False)

    def get_test(self):
        """Returns the testing dataset."""
        return self._make_dataset(self.test_df, shuffle=False)

    def inverse_transform(self, data):
        """
        Applies the inverse transformation to the scaled data.

        Parameters:
            data (np.array): Scaled data to inverse transform.

        Returns:
            np.array: Original scale data.
        """
        return self.scaler.inverse_transform(data)

def extract_data(data):
    """
    Extracts inputs and targets from a dataset.
    
    Parameters:
        data (iterable): An iterable of (inputs, targets).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Concatenated inputs and targets.
    """
    inputs_list = []
    targets_list = []
    for batch_inputs, batch_targets in data:
        inputs_list.append(batch_inputs)
        targets_list.append(batch_targets)
    return np.concatenate(inputs_list, axis=0), np.concatenate(targets_list, axis=0)

def load_data(args):
    """
    Loads or generates training, validation, and testing datasets based on the specified configurations.
    For the 'toy' dataset, it generates synthetic data. For real datasets, it utilizes the TSFDataLoader.

    Parameters:
        args (argparse.Namespace): Command line arguments specifying dataset configurations.

    Returns:
        tuple: Training, validation, and test datasets as tf.data.Dataset objects.
    """
    if args.data == 'toy':
        np.random.seed(args.seed)
        sizes = [10000, 5000, 2000]  # Sizes for train, val, and test
        inputs = [np.random.normal(0, 1, (size, args.seq_len, 7)) for size in sizes]
        W = np.random.normal(0, 1, (args.seq_len, args.pred_len))

        targets = []
        for inputs_i in inputs:
            transposed_inputs = np.transpose(inputs_i, (0, 2, 1))
            linear_targets = np.matmul(transposed_inputs, W)
            noise = np.random.normal(loc=0, scale=1, size=linear_targets.shape)
            noisy_targets = linear_targets + noise
            targets.append(np.transpose(noisy_targets, (0, 2, 1)))

        train_targets, val_targets, test_targets = targets

        train_data = tf.data.Dataset.from_tensor_slices((inputs[0], train_targets)).batch(args.batch_size)
        val_data = tf.data.Dataset.from_tensor_slices((inputs[1], val_targets)).batch(args.batch_size)
        test_data = tf.data.Dataset.from_tensor_slices((inputs[2], test_targets)).batch(args.batch_size)
    else:
        data_loader = TSFDataLoader(args.data, args.batch_size, args.seq_len, args.pred_len, args.feature_type, args.target)
        train_data, val_data, test_data = data_loader.get_train(), data_loader.get_val(), data_loader.get_test()
    return train_data, val_data, test_data
