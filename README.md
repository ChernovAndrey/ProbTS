# BinConv: Binarized Convolutional Forecasting 

**BinConv** is a forecasting model that operates on quantized time series using binarized cumulative encondings representations and convolutional layers. This method demonstrates strong performance on long-term univariate forecasting tasks and is particularly well-suited for quantized probability outputs.

> 🧪 **Forked from [ProbTS](https://github.com/probabilistic-time-series/probts)** – a framework for probabilistic time series forecasting.
For instructions, to setup computation environment, please refer to ProbTS repository.
---

## 🔍 Overview

BinConv introduces a novel approach for time series forecasting by:
- Quantizing continuous values into binary cumulative vectors.
- Applying 2D and 1D convolutions across bins and time dimensions.
- Predicting future values via bin-wise Bernoulli probabilities.

This model  supports both point and probabilistic forecasting setups.

---

## Run Experiments
Run the following scripts to evaluate BinConv and baseline models on the M4 benchmark:
```bash
bash scripts/binconv_m4.sh #for BinConv argmax
```
```bash
bash scripts/binconv_prob_m4.sh # for BinConv sampling
```
```bash
bash scripts/reproduce_m4_results.sh # for the baseline models
```
For long-term mutlivariate datasets:
```bash
bash scripts/reproduce_ltsf_results_binconv_prob.sh #for BinConv Sampling
```

```bash
bash scripts/reproduce_ltsf_96_results.sh # for the baseline models
```
