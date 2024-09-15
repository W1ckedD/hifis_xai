# HFIS XAI

This project introduces a machine learning approach to predict the number of nights a client is likely to spend at homeless shelters within the next six months. It draws inspiration from a previous study conducted in London, Ontario, where researchers used the HIFIS dataset to predict whether a client would experience chronic homelessness within six months, achieving promising results. However, our project aims to enhance that model in the following ways:

Instead of predicting a binary outcome (i.e., whether the client will be chronically homeless in the next six months), we predict the specific number of nights a client is expected to stay in a shelter during that period.

In addition to utilizing the HIFIS dataset, which provides information about a client’s visit history, we incorporated external features such as environmental and economic factors to improve prediction accuracy.

Lastly, we have added model interpretability to explain the reasoning behind each prediction for a specific client.

To use this tool, follow these steps:

1. [Installation and Environment Setup](#installation)
   1. [Required Packages](#required-packages)
   2. [PyTorch](#pytorch)

2. [Configure `config.yaml`](#setup)

3. [Run preprocessing (excluding the most recent 6 months)](#preprocess)

4. [Train the model](#training)

5. [Prepare data for predictions (most recent 6 months)](#inference-preparation)

6. [Make predictions](#prediction)

7. [Run explainability (optional)](#explanations)


## Installation
All experiments were conducted in an Ubuntu 22.04.4 LTS environment with 64GB of RAM, 32 vCPUs, and an RTX 4090 GPU. Python version 3.10.12 and pip version 22.02.2 were used to install the required packages.

### Required Packages
You can install the required packages (excluding PyTorch) using the following command:
```sh
pip install -r requirements.txt
```
Please note that some package versions listed in the `requirements.txt` file may not be available for different Python and pip versions than those specified earlier. While not strictly necessary, it is recommended to install the exact package versions for optimal performance and consistency with the results from this project.
### PyTorch
This project uses PyTorch version 2.4.1 as the primary machine learning framework. Installing the CUDA-enabled version of PyTorch depends on your operating system and the installed CUDA version. It is recommended to install the appropriate PyTorch variant based on your runtime environment. For detailed instructions, refer to the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

## Setup

### Paths

The `PATHS` configuration variable controls the paths and directories for data files and other components required for preprocessing, training, and prediction, as well as those generated during these processes.

If a file is needed for preprocessing, an example file with the expected structure will be provided in the repository.

Please ensure that you set up the required files and configure their paths correctly before running the preprocessing, training, prediction, and explainability modules.


#### Data variables:

<table>
    <tr>
        <td>Variable</td>
        <td>Description</td>
    </tr>
    <tr>
        <td>DATA > RAW > HIFIS</td>
        <td>Path to the raw HIFIS csv file.</td>
    </tr>
    <tr>
        <td>DATA > RAW > LOOKUPS</td>
        <td>Path to the HIFIS lookup xlsx file.</td>
    </tr>
    <tr>
        <td>DATA > EXTERNAL > CPI</td>
        <td>Path to the CPI csv file.</td>
    </tr>
    <tr>
        <td>DATA > EXTERNAL > UNEMPLOYMENT</td>
        <td>Path to the unemployment rate csv file.</td>
    </tr>
    <tr>
        <td>DATA > EXTERNAL > GDP</td>
        <td>Path to the GDP csv file.</td>
    </tr>
    <tr>
        <td>DATA > EXTERNAL > BORDER_CROSSINGS</td>
        <td>Path to the irregular border crossings csv file.</td>
    </tr>
    <tr>
        <td>DATA > EXTERNAL > RENT</td>
        <td>Path to the rent csv file.</td>
    </tr>
    <tr>
        <td>DATA > EXTERNAL > WEATHER</td>
        <td>Path to the weather csv file, including multiple features.</td>
    </tr>
    <tr>
        <td>DATA > PROCESSED</td>
        <td>Path to processed csv file resulted from running the preprocessing module. This path is then also used in training.</td>
    </tr>
    <tr>
        <td>DATA > INFERENCE</td>
        <td>Path to processed csv file resulted from running the infer_process module. This path is then also used during inference and explanation.</td>
    </tr>
</table>

#### Component Variables:

<table>
    <tr>
        <td>Variable</td>
        <td>Description</td>
    </tr>
    <tr>
        <td>COMPONENTS > SCALER</td>
        <td>Path to the scaler component that is saved during training and is then used in prediction and explanation stages</td>
    </tr>
</table>

#### Model Variables

<table>
    <tr>
        <td>Variable</td>
        <td>Description</td>
    </tr>
    <tr>
        <td>MODEL > CHECKPOINTS</td>
        <td>The directory in which the model checkpoints are stored during training.</td>
    </tr>
</table>

#### Explainability Variables

<table>
    <tr>
        <td>Variable</td>
        <td>Description</td>
    </tr>
    <tr>
        <td>EXPLAINABILITY > SHAP > PLOTS</td>
        <td>The directory in which the local explanations of each input instance are stored during the explanation runs.</td>
    </tr>
    <tr>
        <td>EXPLAINABILITY > SHAP > SUMMARY</td>
        <td>Path where the summary plot for the global explanation of the model will be stored</td>
    </tr>
</table>

#### Inference Variables

<table>
    <tr>
        <td>Variable</td>
        <td>Description</td>
    </tr>
    <tr>
        <td>INFERENCE > CLIENT_LIST</td>
        <td>Path to the yml file which holds the list of clients intended for inference / local explanation</td>
    </tr>
    <tr>
        <td>INFERENCE > RESULTS</td>
        <td>Path where the csv file resulted from the inference script is saved. This file includes the ClientHash of the clients as well as the model predeictions for those clients.</td>
    </tr>
</table>

## Preprocess

Before training, the data must be preprocessed to ensure it is suitable for model training. The preprocessing step includes the following:

1. Dropping undesired features and merging the main data with lookup tables.
2. Modifying rows where the client's citizenship status is not "Canadian citizen" or the veteran status is "Canadian Armed Forces."
3. Merging consecutive visits where the client was automatically discharged and immediately readmitted to the shelter.
4. Applying one-hot encoding to categorical features.
5. Implementing the Extending and Sliding Window algorithms.
6. Calculating Ground Truth values.
7. Saving the processed data.


Please note that since the task is to predict the number of nights a client is likely to spend at a shelter within the next 6 months, it is necessary to have at least 6 months of future data on client stays to calculate the ground truth for training. Therefore, the most recent 6 months of HIFIS data cannot be used for training, as the ground truth for this period cannot be determined.

The features to be dropped can be specified in the `PREPROCESS > MERGE_LOOKUPS > COLUMNS_TO_DROP` variable in the `config.yml` file.

Similarly, the HIFIS features to be one-hot encoded can be configured in the `PREPROCESS > ONE_HOT_ENCODING > COLUMNS_TO_ENCODE` variable in the `config.yml` file.


#### Weather Data

The weather data used in this project was collected from the [Visual Crossing Weather API](https://www.visualcrossing.com/weather-api), which provides a range of weather-related features. The numerical features utilized in the preprocessing module can be adjusted in the `PREPROCESS > EXTERNAL_FEATS > WEATHER` section of the `config.yml` file.


#### Sliding Window

The sliding window algorithm is used to calculate and assemble sequences of time-series features. The process begins at the date specified in `PREPROCESS > SLIDING_WINDOW > START_DATE`. A window of size `PREPROCESS > SLIDING_WINDOW > WINDOW_SIZE` (in months) is created, and the time-series features within this window are calculated and assembled into sequences. The window is then slid forward by `PREPROCESS > SLIDING_WINDOW > WINDOW_STEP` (in months). This process continues until 6 months before the latest admission date in the dataset, as explained previously.


#### Extending Window

Static features (non-time-series) are aggregated within a time window starting from the earliest admission date in the dataset up to `PREPROCESS > SLIDING_WINDOW > START_DATE`. This window is then extended by `PREPROCESS > SLIDING_WINDOW > WINDOW_STEP` (in months). Similar to the sliding window, this process is repeated until 6 months before the latest admission date in the dataset.

Note that different aggregation methods are applied to visit-level and client-level static features. You can customize these aggregation methods in the `PREPROCESS > AGGREGATION` section of the `config.yml` file.


#### Ground Truth

The goal of the ML model is to predict the number of nights a client is likely to spend at a shelter 6 months into the future based on their historical data. Therefore, the preprocessed historical data for a client, up to the end of a preprocessing window, is associated with the total number of shelter stays for that client during the 6 months following the end of that window. The duration for which ground truth is calculated can be customized using the `PREPROCESS > GT > DURATION` variable in the `config.yml` file.


#### Parallel Computation

Preprocessing steps must be performed separately for each client and repeated multiple times (for each window), which can be time-consuming. To address this, parallel computing is utilized to leverage multiple processing cores and execute these processes concurrently for multiple clients. You can configure the number of processing units dedicated to this module using the `PREPROCESS > PARALLEL_COMPUTING > N_PROCESS` variable in the `config.yml` file.

**Note:** Ensure that you leave sufficient CPU cores available for background tasks when configuring this variable.

After configuring the necessary settings, you can run the preprocessing script with the following command:

```sh
python -m src.data_processing.preprocess
```

## Training

This project uses [WandB](https://wandb.ai/home) for logging and monitoring the model training process. To create a free WandB account and set up WandB on your local machine, refer to the [WandB Quickstart](https://docs.wandb.ai/quickstart) page.

After configuring WandB, update the project name and experiment name in the `TRAIN > WANDB` variables in the `config.yml` file as needed.


The model consists of a series of MLP and LSTM blocks. Static features are inputs to the MLP layers, while sequences of time-series features are fed directly into the LSTM layers.

Time-series features are generated during preprocessing with a sequence length specified by `SLIDING_WINDOW > WINDOW_SIZE` (in months) and include `'_month_'` in their names. This window size determines the length of the time-series feature sequences. Therefore, if you adjust the window size during preprocessing, it is important to update the `TRAIN > MODEL > TS_SEQ_LEN` variable in the `config.yml` file accordingly.

Additionally, if you add or remove weather-related features during preprocessing, you need to update the number of time-series features in the `TRAIN > MODEL > NUM_TS_FEATS` variable. By default, this variable is set to `12`, which corresponds to the following time-series features: `[stays, cpi, gdp, rent, border_crossing, unemployment_rate, precip, tempmin, snow, snowdepth, windgust, windspeed]`.

#### Chronic Threshold

The primary task of this machine learning model is to predict the number of nights an individual is likely to spend at a shelter within the next 6 months, which is a regression task. However, for evaluation purposes, classification metrics such as `Accuracy`, `Precision`, `Recall`, and `F1 Score` can be useful. To use these metrics, a threshold is required to convert continuous model predictions into binary values, where `1` represents a positive ('chronic') prediction and `0` represents a negative ('non-chronic') prediction.

Given the nature of this task, false negative predictions (where a chronic individual is incorrectly classified as non-chronic) are more critical than false positives. Therefore, the model should focus on reducing false negatives, making a higher `Recall` more desirable than a higher `Precision`. While the core task is regression, which does not allow direct emphasis on recall through loss function adjustments (unlike in the [CHAI-Model](https://www.arxiv.org/pdf/2009.09072)), we can still evaluate the model by adjusting the chronic threshold.

The `TRAIN > MODEL > CHRONIC_THRESHOLD` variable defines this threshold for evaluation purposes. Note that this threshold does not affect model training or output but is used solely for evaluating the model's performance in terms of recall and precision.


#### Training Acceleration

The model training process can be accelerated by utilizing a GPU, although it is not mandatory. Using a GPU is highly recommended for faster and more efficient execution. Regardless of whether a GPU is used, it is crucial that all model parameters and tensors are allocated to the same device. You can switch between `CPU` and `GPU` for all required weights and tensors by adjusting the `TRAIN > ACCELERATOR` variable in the `config.yml` file.

The model leverages `lightning.LightningModule`, and the `lightning.Trainer` module is used in the training script to support parallel training across multiple GPUs through strategies such as `DDP` (Distributed Data Parallel). You can configure this strategy using the `TRAIN > STRATEGY` variable in the `config.yml` file. For more details on supported strategies, please refer to the [Lightning Documentation](https://lightning.ai/docs/pytorch/stable/extensions/strategy.html).


**Note:** When utilizing parallel training strategies like `DDP`, be mindful of adjusting other hyperparameters such as `BATCH_SIZE` and `LEARNING_RATE` to prevent training issues, such as getting stuck in local optima.

Once you have configured all the required parameters, you can run the training script using the following command:

```sh
python -m src.train.single_train
```

### Cross-Validation

To ensure the reliability of the model's results across various time periods, it is advisable to run a cross-validation experiment and use the average model performance for overall evaluation. However, because this is a forecasting problem, traditional cross-validation methods are not suitable. Instead, we employ the nested-K-fold cross-validation technique proposed in the [CHAI-Model](https://github.com/aildnont/HIFIS-model?tab=readme-ov-file#cross-validation) for evaluating model performance across different prediction horizons.

For more details on this cross-validation method, please refer to the [CHAI-Model paper](https://www.arxiv.org/pdf/2009.09072).


The number of iterations or folds for this nested-K-fold cross-validation technique can be configured using the `TRAIN > NESTED_KFOLD > N_ITERS` variable in the `config.yml` file. Once configured, you can run the cross-validation script using the following command:

```sh
python -m src.train.train_cross_val
```

## Inference Preparation

Similar to the training phase, the data must be processed for the inference or prediction stage. The primary difference is that only the most recent data is used for inference. As a result, the concepts of sliding and extending windows are not applicable here. Specifically, the most recent 6 months of data that was excluded during preprocessing and training will be processed for inference.

Please note that no evaluation is performed during the inference stage, so it is not necessary to calculate ground truth values for this stage.


Please note that, like preprocessing, inference processing also benefits from parallel computation. The number of CPU cores dedicated to this task can be configured using the `PREPROCESS > PARALLEL_COMPUTING > N_PROCESS` variable in the `config.yml` file.

For running the inference data processing script, execute the following command:

```sh
python -m src.data_processing.infer_process
```

## Prediction

The prediction script utilizes the previously checkpointed weights of the model to forecast the number of nights a user-specified list of clients is likely to spend in shelters over the next 6 months. This prediction is based on the most recent 6 months of historical data, which should be processed using the inference data processing script.

Before running the prediction script, make sure to:
1. Set the `INFERENCE > CKPT_PATH` to the path of your desired model checkpoints.
2. Set the `INFERENCE > SCALER_PATH` to the path of the scaler that was saved during training.

You can specify the clients for whom you want to make predictions by including their `ClientHash` in a `.yml` file. Set the path to this `.yml` file under `PATHS > INFERENCE > CLIENT_LIST`. The script will then generate predictions for each client in the list and save the results in CSV format under `PATHS > INFERENCE > RESULTS`.


The prediction script can be run by executing the following command:

```sh
python -m src.inference.predict
```

## Explanations

The SHAP (Shapley Additive Explanations) method was used to measure and visualize the impact of the most influential feature inputs on predictions. Explanations are provided at two levels:

1. **Global Level**: This includes collective SHAP values calculated from 2000 randomly selected samples, sorted based on their average influence across the dataset.
2. **Instance Level**: This provides insights into the most impactful features for each individual instance.

Before running the explanation scripts, ensure the following configuration variables are set:
- **Model Checkpoints**: The path to the model checkpoints used for generating explanations is the same as for the inference script. Set this under `INFERENCE > CKPT_PATH`.
- **Scaler**: Set the path to the scaler used during training under `INFERENCE > SCALER_PATH`.


You can run the global shap explanations for the selected checkpoint weights by executing the following command:
```sh
python -m src.explaiability.shap_explain
```
Alternatively, you can average the SHAP values calculated for different model weights. To do this:

1. Specify the directory containing all the desired model weights under `EXPLAINABILITY > SHAP > CKPT_DIR`.
2. Since calculating SHAP values is time-consuming, the script will save the SHAP values as it iterates through the model weights. These values are saved under `EXPLAINABILITY > SHAP > OUTPUT_DIR`.

The script for calculating and averaging SHAP values can be executed as follows:

```sh
python -m src.explainability.shap_global_explain_avg
```
To calculate and visualize single instance explanations for selected clients during the inference stage:

1. Modify the list of selected clients by adding or removing their `ClientHash` in the `.yml` file specified under `PATHS > INFERENCE > CLIENT_LIST`.
2. Configure the `EXPLAINABILITY > SHAP > RUN_NAME` to specify the directory where SHAP’s decision and waterfall plots for each `ClientHash` will be saved.

The script for calculating instance explanations can be run as follows:

```sh
python -m src.explainability.shap_explain_instances
```
