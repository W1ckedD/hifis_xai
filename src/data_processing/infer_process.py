import time
import pandas as pd
from colorama import Fore, Style
from tqdm import tqdm
from src.utils import load_config, log_time

from src.data_processing.preprocess import(
  aggregate_features,
  remove_after_date,
  calculate_client_ts_feats,
  calculate_ts_external_data,
  load_data,
  load_external_data,
  fix_citizenship_veteran,
  merge_lookups,
  one_hot_encode
)

def process_features(
    cfg: dict,
    df: pd.DataFrame,
    external_data: dict[pd.DataFrame],
    cat_feats: list[str],
    numerical_feats: list[str],
    ts_window_size: int,
    ts_step_size: int,
    max_date:pd.Timestamp | str,
  ) -> pd.DataFrame:
  """
  Process the features and calculate the ground truth for a given window of data.

  Args:
    df_window (DataFrame): The DataFrame containing the data for the window.
    external_data (dict): A dictionary containing the external data.
    cat_feats (list): The categorical features to use.
    numerical_feats (list): The numerical features to use.
    ts_window_size (int): The size of the time-series window in months.
    ts_step_size (int): The step size of the time-series window in months.
    max_date (str): The start date of the ground truth window.

  Returns:
    DataFrame: The preprocessed data for the window.
  """

  df = remove_after_date(df, max_date)

  log_time("AGGREGATING FEATURES", 'y')
  df['StayDuration'] = (df['StayDateEnd'] - df['StayDateStart']).dt.days
  df_feats = aggregate_features(cfg, df, ['ClientHash'], cat_feats, numerical_feats)
  
  log_time("CALCULATING CLIENT TIME SERIES FEATURES", 'y')
  df_ts_feats = calculate_client_ts_feats(cfg, df, external_data, max_date, window_size=ts_window_size, step_size=ts_step_size)

  log_time("CALCULATING EXTERNAL TIME SERIES FEATURES", 'y')
  external_ts_feats = calculate_ts_external_data(cfg, external_data, max_date, step_size=ts_step_size, window_size=ts_window_size)
  
  log_time("MERGING ALL TIME SERIES FEATURES", 'y')
  for key, value in tqdm(external_ts_feats.items()):
    df_ts_feats.loc[:, key] = value


  log_time("MERGING STATIC AND TS DATA", 'y')
  df = pd.merge(df_feats, df_ts_feats, on='ClientHash', how='left')
 
  df.insert(0, 'Date', max_date)
  
  return df

def preprocess_infer(cfg: dict) -> pd.DataFrame:
  """
  Processes the HIFIS dataset for inference. This means that all of the data will be processed for aggregating the static features, the data related to the most recent 6 months will be considered for time series calculations, and no GT (ground truth) calculation will be taken place. Since the patterns of the client behavior is already known during training, there is no need for sliding and extending windows when processing during inference. The goal is to create a single row for each client, that includes all of their visit history in HIFIS, as the most recent 6 months of time series sequences. Note that since, "-n_stays" is one of the features in the time series, a client history of minimum 6 months is required for each client for ideal results.

  Args:
    cfg (dict): Configuration dictionary

  Returns:
    pd.DataFrame: Processed data for inference
  """

  # Load the data
  log_time("LOADING DATA")
  main_data, lookup_tables = load_data(cfg)
  external_data = load_external_data(cfg)

  # Fix the citizenship and veteran status
  log_time("FIXING CITIZENSHIP AND VETERAN STATUS")
  main_data = fix_citizenship_veteran(main_data)

  # Merge the lookup tables with the main data
  log_time("MERGING LOOKUP TABLES")
  main_data = merge_lookups(cfg, main_data, lookup_tables)

  main_data['StayDateStart'] = pd.to_datetime(main_data['StayDateStart'])
  main_data['StayDateEnd'] = pd.to_datetime(main_data['StayDateEnd'])

  # one-hot encode the categorical columns
  log_time("ONE-HOT ENCODING")
  cat_columns = cfg['PREPROCESS']['ONE_HOT_ENCODING']['COLUMNS_TO_ENCODE']
  main_data, encoder, cat_feature_names = one_hot_encode(main_data, cat_columns)


  TS_WINDOW_SIZE = cfg['PREPROCESS']['SLIDING_WINDOW']['WINDOW_SIZE']
  TS_STEP_SIZE = cfg['PREPROCESS']['SLIDING_WINDOW']['WINDOW_STEP']
  max_date = pd.to_datetime(main_data['StayDateEnd'].max()) + pd.DateOffset(days=1)

  df = process_features(
    cfg,
    main_data,
    external_data=external_data,
    cat_feats=[*cat_feature_names, 'ReasonForDischarge_Ongoing'],
    numerical_feats=['Age'],
    ts_window_size=TS_WINDOW_SIZE,
    ts_step_size=TS_STEP_SIZE,
    max_date=max_date
  )

  log_time("SAVING DATA", 'c')
  df.reset_index().to_csv(cfg['PATHS']['DATA']['INFERENCE'], index=False)



if __name__ == '__main__':

  time_start = time.time()

  cfg = load_config('config.yml')
  print(f"{Fore.GREEN}PERFORMING PREPROCESSING IN PARALLEL: {cfg['PREPROCESS']['PARALLEL_COMPUTING']['N_PROCESS']} CPU CORES{Style.RESET_ALL} ")
  preprocess_infer(cfg)

  time_end = time.time()
  print(f"{Fore.GREEN}Process completed in:{Style.RESET_ALL} {time.strftime('%H:%M:%S', time.gmtime(time_end - time_start))}")