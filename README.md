# SAMformer

## Overview
This is the official implementation of SAMformer, a novel transformer architecture designed for time series forecasting. It uniquely integrates Sharpness-Aware Minimization (SAM) with a Channel-Wise Attention mechanism. This method provides state-of-the-art performance in multivariate long-term forecasting across various forecasting tasks.

## Installation
To get started with SAMformer, clone this repository and install the required packages.


```bash
git clone https://github.com/romilbert/samformer.git
cd SAMformer
pip install -r requirements.txt
```

Ensure you have Python 3.8 or newer installed.

## Modules
SAMformer consists of several key modules:
- `models/`: Contains the SAMformer model definition along with necessary model components, normalizations, and optimizations.
- `utils/`: Includes utilities for data processing, training, callbacks, and saving the results.
- `dataset/`: Directory for storing datasets used in experiments. Initially, this directory contains only the `ETTh1.csv` dataset for demonstration. You can download all the datasets used in the experiments (ETTh1, ETTh2, ETTm1, ETTm2, electricity, weather, traffic, exchange_rate) [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).

## Usage
To launch the training and evaluation process, use the `run_script.sh` script with the appropriate arguments :
```bash
sh run_script.sh -m [model_name] -d [dataset_name] -s [sequence_length] -u -a
```


### Script Arguments
- `-m`: Model name.
- `-d`: Dataset name.
- `-s`: Sequence length. Default is 512.
- `-u`: Activate Sharpness-Aware Minimization (SAM). Optional.
- `-a`: Activate additional results saving. Optional.

## Example
```bash
sh run_script.sh -m transformer -d ETTh1 -u -a
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
If you have any questions or want to use the code, please contact romain.ilbert@hotmail.fr .

## Acknowledgements 
We would like to express our gratitude to all the researchers and developers whose open-source software has contributed to the development of SAMformer. Special thanks to the developers of Sharpness-Aware Minimization, TSMixer, and Sigma Reparam for their instructive works, which have enabled our approach. Your contributions to the field of machine learning and time series forecasting are greatly appreciated.

