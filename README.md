# SAMformer

## Overview
This is the official implementation of SAMformer, a novel lightweight transformer architecture designed for time series forecasting. It uniquely integrates Sharpness-Aware Minimization (SAM) with a Channel-Wise Attention mechanism. This method provides state-of-the-art performance in multivariate long-term forecasting across various forecasting tasks. In particular, SAMformer surpasses the current state-of-the-art model [TSMixer](https://github.com/google-research/google-research/tree/master/tsmixer/tsmixer_basic) by with a **14.33% relative improvement** on eight benchmarks, while having $\mathbf{\sim4}$ times fewer parameters.

## Architecture
SAMformer takes as input a time series with D dimensions and of length L (*look-back window*), arranged in a matrix (denoted as X, which belongs to R^DxL) and predicts its next H values (*prediction horizon*), denoted by Y $\in$ R^DxH. The main components of the architecture are the following:

💡 **Shallow transformer encoder.** The neural network at the core of SAMformer is a shallow encoder of a simplified [Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). Channel-wise attention is applied to the input, followed by a residual connection. Instead of the usual feedforward block, a linear layer is directly applied on top of the residual connection to output the prediction.

💡 **Channel-Wise Attention.**  Contrary to the usual temporal attention in $\mathbb{R}^{L \times L}$, the channel-wise self-attention is represented by a matrix in $\mathbb{R}^{D \times D}$ and consists of the pairwise correlations between the input's features. This brings two important benefits: 
- Feature permutation invariance, eliminating the need for positional encoding, commonly applied before the attention layer;
- Reduced time and memory complexity as $D \leq l$ in most of the real-world datasets.

💡 **Reversible Instance Normalization (RevIN).** The resulting network is equipped with [RevIN](https://openreview.net/pdf?id=cGDAkQo1C0p), a two-step normalization scheme to handle the shift between the training and testing time series. The official implementation of RevIN is available [here](https://github.com/ts-kim/RevIN).
 
💡 **Sharpness-Aware Minimization (SAM).** As suggested by our empirical and theoretical analysis, we optimize the model with [SAM](https://openreview.net/pdf?id=6Tm1mposlrM) to make it converge towards flatter minima, hence improving its generalization capacity. The official implementation of SAM is available [here](https://github.com/google-research/sam).

SAMformer uniquely combines all these components in a lightweight implementation with very few hyperparameters. We display below the resulting architecture. 

<p align="center">
  <img src="https://github.com/romilbert/samformer/assets/64415312/81b7eef3-f09e-479c-9be4-84fbb66f3aa4" width="200">
</p>


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
Feel free to contact romain.ilbert@hotmail.fr if you have any questions and do not hesitate to open an issue, we would be happy to integrate your suggestions.

## Acknowledgements 
We would like to express our gratitude to all the researchers and developers whose open-source software has contributed to the development of SAMformer. Special thanks to the developers of Sharpness-Aware Minimization, TSMixer, and Sigma Reparam for their instructive works, which have enabled our approach. Your contributions to the field of machine learning and time series forecasting are greatly appreciated.

