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


"""Load raw data and generate time series dataset."""


DATA_DIR = 'gs://time_series_datasets'
LOCAL_CACHE_DIR = './dataset/'


class TSFDataLoader:
  """Generate data loader from raw data."""

  def __init__(
      self, data, batch_size, seq_len, pred_len, feature_type, target='OT'
  ):
    self.data = data
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.feature_type = feature_type
    self.target = target
    self.target_slice = slice(0, None)

    self._read_data()

  def _read_data(self):
    """Load raw data and split datasets."""

    # copy data from cloud storage if not exists
    if not os.path.isdir(LOCAL_CACHE_DIR):
      os.mkdir(LOCAL_CACHE_DIR)

    file_name = self.data + '.csv'
    cache_filepath = os.path.join(LOCAL_CACHE_DIR, file_name)
    if not os.path.isfile(cache_filepath):
      tf.io.gfile.copy(
          os.path.join(DATA_DIR, file_name), cache_filepath, overwrite=True
      )
    df_raw = pd.read_csv(cache_filepath)

    # S: univariate-univariate, M: multivariate-multivariate, MS:
    # multivariate-univariate
    df = df_raw.set_index('date')
    if self.feature_type == 'S':
      df = df[[self.target]]
    elif self.feature_type == 'MS':
      target_idx = df.columns.get_loc(self.target)
      self.target_slice = slice(target_idx, target_idx + 1)

    # split train/valid/test
    n = len(df)
    if self.data.startswith('ETTm'):
      train_end = 12 * 30 * 24 * 4
      val_end = train_end + 4 * 30 * 24 * 4
      test_end = val_end + 4 * 30 * 24 * 4
    elif self.data.startswith('ETTh'):
      train_end = 12 * 30 * 24
      val_end = train_end + 4 * 30 * 24
      test_end = val_end + 4 * 30 * 24
    else:
      train_end = int(n * 0.7)
      val_end = n - int(n * 0.2)
      test_end = n
    train_df = df[:train_end]
    val_df = df[train_end - self.seq_len : val_end]
    test_df = df[val_end - self.seq_len : test_end]

    # standardize by training set
    self.scaler = StandardScaler()
    self.scaler.fit(train_df.values)

    def scale_df(df, scaler):
      data = scaler.transform(df.values)
      return pd.DataFrame(data, index=df.index, columns=df.columns)

    self.train_df = scale_df(train_df, self.scaler)
    self.val_df = scale_df(val_df, self.scaler)
    self.test_df = scale_df(test_df, self.scaler)
    self.n_feature = self.train_df.shape[-1]

  def _split_window(self, data):
    inputs = data[:, : self.seq_len, :]
    labels = data[:, self.seq_len :, self.target_slice]
    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.seq_len, None])
    labels.set_shape([None, self.pred_len, None])
    return inputs, labels

  def _make_dataset(self, data, shuffle=True):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=(self.seq_len + self.pred_len),
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=self.batch_size,
    )
    ds = ds.map(self._split_window)
    return ds

  def inverse_transform(self, data):
    return self.scaler.inverse_transform(data)

  def get_train(self, shuffle=True):
    return self._make_dataset(self.train_df, shuffle=shuffle)

  def get_val(self):
    return self._make_dataset(self.val_df, shuffle=False)

  def get_test(self):
    return self._make_dataset(self.test_df, shuffle=False)

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

        return train_data, val_data, test_data
    else:
        data_loader = TSFDataLoader(args.data, args.batch_size, args.seq_len, args.pred_len, args.feature_type, args.target)
        train_data, val_data, test_data = data_loader.get_train(), data_loader.get_val(), data_loader.get_test()

        return train_data, val_data, test_data, data_loader.n_feature
