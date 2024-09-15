import os
import joblib
import shap
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import lightning as pl
import matplotlib.pyplot as plt
from src.model.model import ModelRegression
from src.utils import load_config, log_time

def load_background_data(cfg: dict) -> pd.DataFrame:
  """
  Load the background data from the given path.

  Args:
    cfg (dict): The configuration dictionary.

  Returns:
    data (pd.DataFrame): The background data loaded from the given path.
  """
  data = pd.read_csv(cfg['PATHS']['DATA']['PROCESSED'])
  data = data.drop(columns=['GT', 'GT_binary', 'ReasonForDischarge_Ongoing', 'ClientHash', 'Date'])
  return data

def load_inference_data(cfg: dict) -> pd.DataFrame:
  """
  Load the data from the given path.

  Args:
    cfg (dict): The configuration dictionary.

  Returns:
    data (pd.DataFrame): The data loaded from the given path.
  """
  data = pd.read_csv(cfg['PATHS']['DATA']['INFERENCE'])
  data = data.drop(columns=['ReasonForDischarge_Ongoing', 'ClientHash', 'Date'])
  return data

def load_scaler(cfg: dict) -> joblib:
  """
  Load the scaler from the given path.

  Args:
    cfg (dict): The configuration dictionary.

  Returns:

  """
  scaler = joblib.load(cfg['INFERENCE']['SCALER_PATH'])
  return scaler

def load_model(cfg: dict, x_static_input_dim: int) -> pl.LightningModule:
  """
  Load the model from the given path.

  Args:
    cfg (dict): The configuration dictionary.
    x_static_input_dim (int): The number of static input features.

  Returns:
    model (pl.LightningModule): The model loaded from the given path.
  """
  

  hidden_dim = cfg['TRAIN']['MODEL']['MLP_HIDDEN_DIM']
  n_hidden = cfg['TRAIN']['MODEL']['MLP_N_HIDDEN']
  output_dim = 1
  dropout_p = cfg['TRAIN']['MODEL']['DROPOUT_PROB']

  lstm_hidden_dim = cfg['TRAIN']['MODEL']['LSTM_HIDDEN_DIM']
  n_lstm_hidden = cfg['TRAIN']['MODEL']['LSTM_N_HIDDEN']
  n_ts_feat = cfg['TRAIN']['MODEL']['NUM_TS_FEATS']
  ts_seq_len = cfg['TRAIN']['MODEL']['TS_SEQ_LEN']

  chronic_threshold = cfg['TRAIN']['MODEL']['CHRONIC_THRESHOLD']

  ckpt_path = cfg['INFERENCE']['CKPT_PATH']

  model = ModelRegression.load_from_checkpoint(
    ckpt_path,
    input_dim=x_static_input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    n_hidden=n_hidden,
    dropout_p=dropout_p,
    n_ts_feat=n_ts_feat,
    seq_len=ts_seq_len,
    n_lstm_hidden=n_lstm_hidden,
    lstm_hidden_dim=lstm_hidden_dim,
    chronic_threshold=chronic_threshold
  )

  return model

def get_static_feature_names(data: pd.DataFrame) -> list:
  """
  Get the static feature names from the given data.

  Args:
    data (pd.DataFrame): The data from which the static feature names are to be extracted.

  Returns:
    static_feature_names (list): The list of static feature names.
  """
  static_feature_names = [feat for feat in data.columns if '_month_' not in feat]
  return static_feature_names

def get_static_feature_indices(data: pd.DataFrame, static_feature_names: list) -> list:
  """
  Get the static feature indices from the given data.

  Args:
    data (pd.DataFrame): The data from which the static feature indices are to be extracted.
    static_feature_names (list): The list of static feature names.

  Returns:
    static_feature_indices (list): The list of static feature indices.
  """
  static_feature_indices = [data.columns.get_loc(feat) for feat in static_feature_names]
  return static_feature_indices

def get_time_series_feature_indices(data: pd.DataFrame, time_series_feature_names: list) -> list:
  """
  Get the time series feature indices from the given data.

  Args:
    data (pd.DataFrame): The data from which the time series feature indices are to be extracted.
    time_series_feature_names (list): The list of time series feature names.

  Returns:
    time_series_feature_indices (list): The list of time series feature indices.
  """
  time_series_feature_indices = [data.columns.get_loc(feat) for feat in time_series_feature_names]
  return time_series_feature_indices


def get_time_series_feature_names(data: pd.DataFrame) -> list:
  """
  Get the time series feature names from the given data.

  Args:
    data (pd.DataFrame): The data from which the time series feature names are to be extracted.

  Returns:
    time_series_feature_names (list): The list of time series feature names.
  """
  time_series_feature_names = [feat for feat in data.columns if '_month_' in feat]
  time_series_feature_names = sorted(time_series_feature_names, key=lambda x: int(x.split('_month_')[0]))
  return list(time_series_feature_names)

def explain_model_summary() -> None:
  log_time('Loading the configuration file')
  cfg = load_config('config.yml')
  ACCELERATOR = cfg['TRAIN']['ACCELERATOR']
  device = torch.device('cuda' if ACCELERATOR == 'GPU' else 'cpu')
  
  # Load the data
  log_time('Loading the data')
  data = load_background_data(cfg)
  # Get Static Feature Names
  static_feature_names = get_static_feature_names(data)
  # Get Time Series Feature Names
  time_series_feature_names = get_time_series_feature_names(data)
  # Get Static Feature Indices
  static_feature_indices = get_static_feature_indices(data, static_feature_names)
  # Get Time Series Feature Indices
  time_series_feature_indices = get_time_series_feature_indices(data, time_series_feature_names)


  log_time('Loading the scaler')
  # Load the scaler
  scaler = load_scaler(cfg)
  # Load the model
  log_time('Loading the model')
  model = load_model(cfg, len(static_feature_names))
  model.eval().to(device)

  log_time('Scaling the data')
  # Scale the data
  skip_cols = ['GT', 'GT_binary', 'sum_stays', 'ClientHash', 'ReasonForDischarge_Ongoing', 'Date']
  scale_cols = data.columns.difference(skip_cols)
  data[scale_cols] = scaler.transform(data[scale_cols])

  log_time('Taking K means of background data')
  # Take K means of data
  background_data = shap.kmeans(data.values, 10)
  # Define the pred_fn
  def pred_fn(data: np.ndarray):
    static_feat = data[:, static_feature_indices]
    ts_feat = data[:, time_series_feature_indices]
    static_feat = torch.tensor(static_feat, dtype=torch.float32).to(device)
    ts_feat = torch.tensor(ts_feat, dtype=torch.float32).to(device)

    with torch.no_grad():
      output = model(static_feat, ts_feat)

    return output.cpu().numpy().flatten()
  
  exp = shap.KernelExplainer(pred_fn, background_data)

  sample_data = data.sample(2000, random_state=42)
  print("Housed Rate: ", sample_data[sample_data['Housed'] == sample_data['Housed'].max()].shape[0] / sample_data.shape[0])
  print("ReasonForDischarge_Housed - Subsizied Rate: ", sample_data[sample_data['ReasonForDischarge_Housed - Subsidized'] == sample_data['ReasonForDischarge_Housed - Subsidized'].max()].shape[0] / sample_data.shape[0])
  print("ReasonForDischarge_Housed - Private Market Rate: ", sample_data[sample_data['ReasonForDischarge_Housed - Private Market'] == sample_data['ReasonForDischarge_Housed - Private Market'].max()].shape[0] / sample_data.shape[0])
  # Explain sample data
  log_time('Calculating SHAP values')
  shap_values = exp.shap_values(sample_data.values)
  # Save the summary plot
  log_time('Saving the summary plot')
  print(shap_values.dtype)
  print(shap_values.shape)
  print(sample_data.shape)
  shap.summary_plot(
    shap_values,
    sample_data,
    plot_type='violin',
    plot_size=(12, 10),
    max_display=20,
  )

  os.makedirs(cfg['PATHS']['EXPLAINABILITY']['SHAP']['PLOTS'], exist_ok=True)
  plt.savefig(cfg['PATHS']['EXPLAINABILITY']['SHAP']['SUMMARY'])

if __name__ == '__main__':
  pl.seed_everything(42)
  explain_model_summary()
