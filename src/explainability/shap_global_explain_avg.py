import os
import joblib
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.model.model import ModelRegression
from src.utils import load_config, log_time
from argparse import ArgumentParser

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


def load_scaler(cfg: dict) -> joblib:
  """
  Load the scaler from the given path.

  Args:
    cfg (dict): The configuration dictionary.

  Returns:

  """
  scaler = joblib.load(cfg['INFERENCE']['SCALER_PATH'])
  return scaler

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

def create_pred_fn(
  cfg: dict,
  ckpt_path: str,
  static_feature_indices: list,
  time_series_feature_indices: list
) -> callable:
  """
  Create a prediction function from the given checkpoint path and configuration.

  Args:
    cfg (dict): The configuration dictionary.
    ckpt_path (str): The checkpoint path.
    static_feature_indices (list): The list of static feature indices.
    time_series_feature_indices (list): The list of time series feature indices.

  Returns:
    pred_fn (callable): The prediction
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
  ACCELERATOR = cfg['TRAIN']['ACCELERATOR']

  device = torch.device('cuda' if ACCELERATOR == 'GPU' else 'cpu')

  model = ModelRegression.load_from_checkpoint(
    ckpt_path,
    input_dim=len(static_feature_indices),
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

  model.eval().to(device)

  def pred_fn(data: np.ndarray):
    static_feat = data[:, static_feature_indices]
    ts_feat = data[:, time_series_feature_indices]
    static_feat = torch.tensor(static_feat, dtype=torch.float32).to(device)
    ts_feat = torch.tensor(ts_feat, dtype=torch.float32).to(device)

    with torch.no_grad():
      output = model(static_feat, ts_feat)

    return output.cpu().numpy().flatten()

  return pred_fn
  

def calculate_shap_values(pred_fn: callable, background_data: pd.DataFrame, inference_data: pd.DataFrame) -> np.ndarray:
  """
  Calculate the SHAP values for the given inference data.

  Args:
    pred_fn (callable): The prediction function.
    background_data (pd.DataFrame): The background data.
    inference_data (pd.DataFrame): The inference data.

  Returns:
    shap_values (np.ndarray): The SHAP values.
  """
  
  explainer = shap.KernelExplainer(pred_fn, background_data)
  shap_values = explainer.shap_values(inference_data)

  return shap_values

def save_shap_values(shap_values: np.ndarray, output_path: str):
  """
  Save the SHAP values to the given output path.

  Args:
    shap_values (np.ndarray): The SHAP values.
    output_path (str): The output path.
  """
  np.save(output_path, shap_values)

def avg_summary_plot(shap_values_list: list[np.ndarray], data: pd.DataFrame, output_path: str):
  """
  Create the average SHAP summary plot.

  Args:
    shap_values_list (list[np.ndarray]): The list of SHAP values.
    data (pd.DataFrame): The data.
    output_path (str): The output path.
  """

  shap_values = np.array(shap_values_list, dtype=np.float64)
  shap_values = np.mean(shap_values, axis=0).astype(np.float64)
  plt.clf()
  shap.summary_plot(shap_values, data, plot_type='violin', plot_size=(12, 10))
  plt.savefig(output_path)  

def main(shap_values_dir: str | None = None) -> None:
  cfg = load_config()
  
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

  skip_cols = ['GT', 'GT_binary', 'sum_stays', 'ClientHash', 'ReasonForDischarge_Ongoing', 'Date']
  scale_cols = data.columns.difference(skip_cols)
  data[scale_cols] = scaler.transform(data[scale_cols])

  log_time('Taking K means of background data')
  # Take K means of data
  background_data = shap.kmeans(data.values, 10)

  sample_data = data.sample(2000, random_state=42)

  ckpt_paths = list(sorted(os.listdir(cfg['EXPLAINABILITY']['SHAP']['CKPT_DIR'])))
  
  if shap_values_dir is not None:
    shap_values_list = []
    with os.scandir(shap_values_dir) as entries:
      for entry in entries:
        if entry.is_file():
          shap_values = np.load(entry.path)
          shap_values_list.append(shap_values)

    log_time('Creating average SHAP summary plot')
    avg_summary_plot(shap_values_list, sample_data, cfg['EXPLAINABILITY']['SHAP']['AVG_SUMMARY_PLOT'])
    return

  shap_values_list = []

  for ckpt in ckpt_paths:
    log_time(f'Creating prediction function for checkpoint: {ckpt}')
    
    ckpt_path = os.path.join(cfg['EXPLAINABILITY']['SHAP']['CKPT_DIR'], ckpt)
    pred_fn = create_pred_fn(cfg, ckpt_path, static_feature_indices, time_series_feature_indices)

    log_time(f'Calculating SHAP values for checkpoint: {ckpt}')
    shap_values = calculate_shap_values(pred_fn, background_data, sample_data)
    shap_values_list.append(shap_values)
    output_dir = os.path.join(cfg['EXPLAINABILITY']['SHAP']['OUTPUT_DIR'])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'shap_values_{ckpt.replace(".ckpt", ".npy")}')
    log_time(f'Saving SHAP values for checkpoint: {ckpt}')
    save_shap_values(shap_values, output_path)

  log_time('Creating average SHAP summary plot')
  avg_summary_plot(shap_values_list, sample_data, cfg['EXPLAINABILITY']['SHAP']['AVG_SUMMARY_PLOT'])

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--shap_values_dir', type=str, default=None)
  args = parser.parse_args()
  main(args.shap_values_dir)
