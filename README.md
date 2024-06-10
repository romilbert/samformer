# SAMformer
**The repository contains the official implementation of SAMformer, a transformer-based model for time series forecasting described in** 

>Romain Ilbert*, Ambroise Odonnat*, Vasilii Feofanov, Aladin Virmaux, Giuseppe Paolo, Themis Palpanas, Ievgen redko.
[SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention](https://arxiv.org/abs/2402.10198). **Oral ICML 2024**.
<br/>*Equal contribution

## Overview
**SAMformer** is a novel lightweight transformer architecture designed for time series forecasting. It uniquely integrates Sharpness-Aware Minimization (SAM) with a Channel-Wise Attention mechanism. This method provides state-of-the-art performance in multivariate long-term forecasting across various forecasting tasks. In particular, SAMformer surpasses the current state-of-the-art model [TSMixer](https://openreview.net/pdf?id=wbpxTuXgm0) by $\mathbf{14.33}$% on average, while having $\mathbf{\sim4}$ times fewer parameters.

## Architecture
**SAMformer** takes as input a $D$-dimensional time series of length $L$ (*look-back window*), arranged in a matrix $\mathbf{X}\in\mathbb{R}^{D\times L}$ and predicts its next $H$ values (*prediction horizon*), denoted by $\mathbf{Y}\in\mathbb{R}^{D\times H}$. The main components of the architecture are the following. 

💡 **Shallow transformer encoder.** The neural network at the core of SAMformer is a shallow encoder of a simplified [Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). Channel-wise attention is applied to the input, followed by a residual connection. Instead of the usual feedforward block, a linear layer is directly applied on top of the residual connection to output the prediction.

💡 **Channel-Wise Attention.**  Contrary to the usual temporal attention in $\mathbb{R}^{L \times L}$, the channel-wise self-attention is represented by a matrix in $\mathbb{R}^{D \times D}$ and consists of the pairwise correlations between the input's features. This brings two important benefits: 
- Feature permutation invariance, eliminating the need for positional encoding, commonly applied before the attention layer;
- Reduced time and memory complexity as $D \leq L$ in most of the real-world datasets.

💡 **Reversible Instance Normalization (RevIN).** The resulting network is equipped with [RevIN](https://openreview.net/pdf?id=cGDAkQo1C0p), a two-step normalization scheme to handle the shift between the training and testing time series.
 
💡 **Sharpness-Aware Minimization (SAM).** As suggested by our empirical and theoretical analysis, we optimize the model with [SAM](https://openreview.net/pdf?id=6Tm1mposlrM) to make it converge towards flatter minima, hence improving its generalization capacity. 

SAMformer uniquely combines all these components in a lightweight implementation with very few hyperparameters. We display below the resulting architecture. 

<p align="center">
  <img src="https://github.com/romilbert/samformer/assets/64415312/81b7eef3-f09e-479c-9be4-84fbb66f3aa4" width="200">
</p>


## Results
We conduct our experiments on various multivariate time series forecasting benchmarks. 

🥇 **Improved performance.** SAMformer outperforms its competitors in $\mathbf{7}$ **out of** $\mathbf{8}$ datasets by a large margin. In particular, it improves over its best competitor TSMixer+SAM by $\mathbf{5.25}$%, surpasses the standalone TSMixer by $\mathbf{14.33}$%, and the best transformer-based model FEDformer by $\mathbf{12.36}$%. In addition, it improves over the vanilla Transformer by $\mathbf{16.96}$%. For each dataset and horizon, SAMformer is ranked either first or second.
<p align="center">
  <img src="https://github.com/romilbert/samformer/assets/64415312/d39ae38e-5f88-47e6-ba5d-04d4de8f2aca" width="600">
</p>

🚀 **Computational efficiency and versatility.** SAMformer has a lightweight implementation with few learnable parameters, contrary to most of its competitors, leading to improved computational efficiency. SAMformer significantly outperforms the SOTA in multivariate time series despite having fewer parameters. In addition, the same architecture is used for all the datasets, while most of the other baselines require heavy hyperparameter tuning, which showcases the versatility of our approach.

📚 **Qualitative benefits.** We display in our paper the benefits of SAMformer in terms of smoothness of the loss landscape, robustness to the prediction horizons, and signal propagation in the attention layer.

## Installation
To get started with SAMformer, clone this repository and install the required packages.


```bash
git clone https://github.com/romilbert/samformer.git
cd SAMformer
pip install -r requirements.txt
```

Make sure you have Python 3.8 or a newer version installed.

## Modules
SAMformer consists of several key modules:
- `models/`: Contains the SAMformer architecture along with necessary components for normalization and optimization.
- `utils/`: Contains the utilities for data processing, training, callbacks, and to save the results.
- `dataset/`: Directory for storing the datasets used in experiments. For illustration purposes, this directory only contains the ETTh1 dataset in .csv format. You can download all the datasets used in our experiments (ETTh1, ETTh2, ETTm1, ETTm2, electricity, weather, traffic, exchange_rate) [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).

## Usage
To launch the training and evaluation process, use the `run_script.sh` script with the appropriate arguments :
```bash
sh run_script.sh -m [model_name] -d [dataset_name] -s [sequence_length] -u -a
```

### Script Arguments
- `-m`: Model name.
- `-d`: Dataset name.
- `-s`: Sequence length. The default is 512.
- `-u`: Activate Sharpness-Aware Minimization (SAM). Optional.
- `-a`: Activate additional results saving. Optional.

## Example
```bash
sh run_script.sh -m transformer -d ETTh1 -u -a
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Open-source Participation
Do not hesitate to contribute to this project by submitting pull requests or issues, we would be happy to receive feedback and integrate your suggestions.

## Contact
- [Romain Ilbert](https://romilbert.github.io/)
- [Ambroise Odonnat](https://ambroiseodt.github.io/)
- [Vasilii Feofanov](https://scholar.google.com/citations?user=UIteS6oAAAAJ&hl=en)
- [Aladin Virmaux](https://avirmaux.github.io/)
- [Giuseppe Paolo](https://www.giupaolo.com/)
- [Themis Palpanas](https://helios2.mi.parisdescartes.fr/~themisp/)
- [Ievgen Redko](https://ievred.github.io/). 

Feel free to contact Romain Ilbert <romain.ilbert@hotmail.fr> - Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com> in case of questions.

## Acknowledgements 
We would like to express our gratitude to all the researchers and developers whose work and open-source software have contributed to the development of SAMformer. Special thanks to the authors of [SAM](https://openreview.net/pdf?id=6Tm1mposlrM), [TSMixer](https://openreview.net/pdf?id=wbpxTuXgm0), [RevIN](https://openreview.net/pdf?id=cGDAkQo1C0p) and $\sigma$[Reparam](https://proceedings.mlr.press/v202/zhai23a/zhai23a.pdf) for their instructive works, which have enabled our approach. We provide below a non-exhaustive list of GitHub repositories that helped with valuable code base and datasets: 
 - [SAM](https://github.com/google-research/sam)
 - [TSMixer](https://github.com/google-research/google-research/tree/master/tsmixer)
 - [RevIN](https://github.com/ts-kim/RevIN)
 - [Informer](https://github.com/zhouhaoyi/Informer2020)
 - [FEDformer](https://github.com/MAZiqing/FEDformer)

