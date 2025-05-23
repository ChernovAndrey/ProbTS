# BinConv: Binarized Convolutional Forecasting 

**BinConv** is a forecasting model that operates on quantized time series using binarized cumulative encondings representations and convolutional layers. This method demonstrates strong performance on long-term univariate forecasting tasks and is particularly well-suited for quantized probability outputs.

> 🧪 **Forked from [ProbTS](https://github.com/microsoft/ProbTS)** – a framework for probabilistic time series forecasting.
For instructions, to setup computation environment, please refer to ProbTS repository.
---

## 🔍 Overview

BinConv introduces a novel approach for time series forecasting by:
- Quantizing continuous values into binary cumulative vectors.
- Applying 2D and 1D convolutions across bins and time dimensions.
- Predicting future values via bin-wise Bernoulli probabilities.

This model  supports both point and probabilistic forecasting setups.

---

## Installation 

### Environment

To set up the environment:

```bash
# Create a new conda environment
conda create -n probts python=3.10
conda activate probts

# Install required packages
pip install .
pip uninstall -y probts # recommended to uninstall the root package (optional)
```

### Datasets
**Univariate Forecasting**: For univariate forecasting, datasets are downloaded automatically during the execution of the bash scripts (see the next section). 

**Multivaraite Forecasting**: To download the [multivariate forecasting datasets](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy), please follow these steps:

```bash
bash scripts/prepare_datasets.sh "./datasets"
```

Configure the datasets using `--data.data_manager.init_args.dataset {DATASET_NAME}` with the following list of available datasets:
```bash
['etth1', 'etth2','ettm1','ettm2','traffic_ltsf', 'electricity_ltsf', 'exchange_ltsf', 'illness_ltsf', 'weather_ltsf', 'caiso', 'nordpool']
```
*Note: When utilizing long-term forecasting datasets, you must explicitly specify the `context_length` and `prediction_length` parameters. For example, to set a context length of 96 and a prediction length of 192, use the following command-line arguments:*
```bash
--data.data_manager.init_args.context_length 96 \
--data.data_manager.init_args.prediction_length 192 \
```
  
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
