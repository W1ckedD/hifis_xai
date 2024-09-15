import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from colorama import Fore, Style
from multiprocesspandas import applyparallel
from src.utils import load_config, log_time


cfg = load_config('config.yml')


def load_data(cfg: dict) -> tuple[pd.DataFrame, dict[pd.DataFrame]]:
  """
  Load the main data and lookup tables from the HIFIS dataset.

  Returns:
    main_data (pandas.DataFrame): The main data loaded from 'raw/HIFIS/HIFIS-Clients.csv'.
    lookup_tables (dict): A dictionary containing the lookup tables loaded from 'raw/HIFIS/HIFIS-Clients-Lookups.xlsx'.
                The keys of the dictionary correspond to the sheet names in the Excel file.
  """
  # Load the data
  main_data = pd.read_csv(cfg['PATHS']['DATA']['RAW']['HIFIS'], parse_dates=['StayDateStart', 'StayDateEnd', 'DOB'], low_memory=False)

  # read all of the sheets in the excel file
  lookups = pd.read_excel(cfg['PATHS']['DATA']['RAW']['LOOKUPS'], sheet_name=None)

  # Extract the lookup tables
  lookup_tables = {}
  for key in lookups.keys():
    lookup_tables[key] = lookups[key]

  return main_data, lookup_tables

def merge_lookups(cfg: dict, main_data: pd.DataFrame, lookup_tables: dict[pd.DataFrame]) -> pd.DataFrame:
  """
  Merge the lookup tables with the main data.

  Args:
    main_data (DataFrame): The main data to be merged with the lookup tables.
    lookup_tables (dict): A dictionary containing the lookup tables.

  Returns:
    DataFrame: The main data with the lookup tables merged.

  """
  main_data = main_data.drop(columns=cfg['PREPROCESS']['MEREGE_LOOKUPS']['COLUMNS_TO_DROP'])

  main_data['StayDateStart'] = pd.to_datetime(main_data['StayDateStart'])
  main_data['StayDateEnd'] = pd.to_datetime(main_data['StayDateEnd'])
  
  # set the ongoing stays to the training start date
  main_data['StayDateEnd'] = main_data['StayDateEnd'].fillna(pd.to_datetime('2023-09-01'))

  # set the chronic status to 1 if the client has a chronic condition
  main_data['HasFamily'] = main_data['FamilyHash'].notnull()
  main_data = main_data.drop(columns=['FamilyHash'])

  # set the current age to the age at intake
  main_data['DOB'] = pd.to_datetime(main_data['DOB'])
  main_data['Age'] = main_data['StayDateStart'].dt.year - main_data['DOB'].dt.year

  main_data = main_data.merge(lookup_tables['Gender'][['ID', 'NameE']], how='left', left_on='GenderID', right_on='ID')
  main_data = main_data.rename(columns={'NameE': 'Gender'})
  main_data = main_data.drop(columns=['GenderID', 'ID'])

  main_data = main_data.merge(lookup_tables['Aboriginal'][['ID', 'NameE']], how='left', left_on='AboriginalIndicatorID', right_on='ID')
  main_data = main_data.rename(columns={'NameE': 'AboriginalIndicator'})
  main_data = main_data.drop(columns=['AboriginalIndicatorID', 'ID'])

  main_data = main_data.merge(lookup_tables['Veteran'][['ID', 'NameE']], how='left', left_on='VeteranStatusID', right_on='ID')
  main_data = main_data.rename(columns={'NameE': 'VeteranStatus'})
  main_data = main_data.drop(columns=['VeteranStatusID', 'ID'])

  main_data = main_data.merge(lookup_tables['Citizenship'][['ID', 'NameE']], how='left', left_on='CitizenshipID', right_on='ID')
  main_data = main_data.rename(columns={'NameE': 'Citizenship'})
  main_data = main_data.drop(columns=['CitizenshipID', 'ID'])

  return main_data

def load_external_data(cfg: dict) -> dict[pd.DataFrame]:
  """
  Load the external data from the HIFIS dataset.

  Args:
    cfg (dict): A dictionary containing the configuration parameters.

  Returns:
    external_data (dict[pandas.DataFrame]): A dictionary containing the external data loaded from 'data/raw/HIFIS/external'.
  """

  # Weather data
  weather = pd.read_csv(cfg['PATHS']['DATA']['RAW']['EXTERNAL']['WEATHER'])
  weather['datetime'] = pd.to_datetime(weather['datetime'])

  # cpi data
  cpi = pd.read_csv(cfg['PATHS']['DATA']['RAW']['EXTERNAL']['CPI'])
  cpi = cpi[cpi['Products and product groups'] == 'All-items']
  cpi['REF_DATE'] = pd.to_datetime(cpi['REF_DATE'])

  # border crossings
  def extend_border_crossing(df_bc: pd.DataFrame) -> pd.DataFrame:
    """
    Extend the border crossing data by creating new rows for each month between the 'DateStart' and 'DateEnd' columns.

    Args:
      df_bc (pd.DataFrame): The input DataFrame containing border crossing data.

    Returns:
      pd.DataFrame: The extended DataFrame with new rows for each month between 'DateStart' and 'DateEnd'.
    """
    df_bc['DateStart'] = pd.to_datetime(df_bc['DateStart'], format='%Y-%m')
    df_bc['DateEnd'] = pd.to_datetime(df_bc['DateEnd'], format='%Y-%m')
    df_bc = df_bc.drop(0)
    new_rows = []
    for index, row in df_bc.iterrows():
      for date in pd.date_range(row['DateStart'], row['DateEnd'], freq='MS'):
        new_row = row.copy()
        new_row['Date'] = str(date.strftime('%Y-%m'))
        new_rows.append(new_row)

    df_bc = pd.DataFrame(new_rows)
    df_bc = df_bc.drop(columns=['DateStart', 'DateEnd', 'Unnamed: 0'])

    return df_bc
  border_crossings = pd.read_csv(cfg['PATHS']['DATA']['RAW']['EXTERNAL']['BORDER_CROSSINGS'])
  border_crossings = extend_border_crossing(border_crossings)
  border_crossings['Date'] = pd.to_datetime(border_crossings['Date'], format='%Y-%m')

  # GDP data
  gdp = pd.read_csv(cfg['PATHS']['DATA']['RAW']['EXTERNAL']['GDP'])
  gdp = gdp[gdp['North American Industry Classification System (NAICS)'] == 'All industries [T001]']
  gdp['REF_DATE'] = pd.to_datetime(gdp['REF_DATE'])

  # rent data
  rent = pd.read_csv(cfg['PATHS']['DATA']['RAW']['EXTERNAL']['RENT'])
  rent = rent[rent['Type of structure'] == 'Row and apartment structures of three units and over']
  rent['REF_DATE'] = pd.to_datetime(rent['REF_DATE'], format='%Y')

  # unemployment data
  unemployment = pd.read_csv(cfg['PATHS']['DATA']['RAW']['EXTERNAL']['UNEMPLOYMENT'])
  unemployment = unemployment[unemployment['Labour force characteristics'] == 'Unemployment rate']
  unemployment = unemployment[unemployment['Population centre and rural areas'] == 'Total, all population centres and rural areas']
  unemployment['REF_DATE'] = pd.to_datetime(unemployment['REF_DATE'])

  external_data = {
    'weather': weather,
    'cpi': cpi,
    'border_crossings': border_crossings,
    'gdp': gdp,
    'rent': rent,
    'unemployment': unemployment
  }

  return external_data

def fix_citizenship_veteran(df: pd.DataFrame) -> pd.DataFrame:
  """
  There have been cases where a client reported their veteran status as 'veteran-Canadian armed forces;
  while not having a Canadian citizenship.

  Args:
    df (pd.DataFrame): input dataframe
  Returns:
    pd.DataFrame: the modified dataframe
  """

  condition = (df['VeteranStatusID'] == 1 & ~df['CitizenshipID'].isin([1, 2]))
  df.loc[condition, 'VeteranStatusID'] = 2
  return df

def remove_after_date(df: pd.DataFrame, date: str) -> pd.DataFrame:
  """
  Remove rows from the dataframe that are after the given date.
  Clip the ongoing stays to the given date.
  Update Age to the age at the given date.

  Args:
    df (DataFrame): The dataframe to remove rows from.
    date (str): The date to remove rows after.

  Returns:
    DataFrame: The dataframe with the rows removed.
  """
  df = df.copy()

  date = pd.to_datetime(date)
  df = df[df['StayDateStart'] <= date]
  df = update_reason_for_discharge(df, date)
  df['StayDateEnd'] = df['StayDateEnd'].clip(upper=date)

  df['Age'] = date.year - df['DOB'].dt.year

  return df

def clip_df(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
  """
  Clip a DataFrame to a given date range.

  Args:
    df (DataFrame): The DataFrame to clip.
    start_date (str): The start date of the clip.
    end_date (str): The end date of the clip.

  Returns:
    DataFrame: The clipped DataFrame.
  """
  df = df.copy()
  condition = (df['StayDateStart'] < end_date) & (df['StayDateEnd'] > start_date)
  df = df[condition]
  df['StayDateStart'] = df['StayDateStart'].clip(lower=start_date)
  df['StayDateEnd'] = df['StayDateEnd'].clip(upper=end_date)

  return df

def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, OneHotEncoder, list[str]]:
  """
  One-hot encode the specified columns in a DataFrame.

  Args:
    df (DataFrame): The DataFrame to one-hot encode.
    columns (list): A list of column names to one-hot encode.

  Returns:
    DataFrame: The DataFrame with the specified columns one-hot encoded.
  """
  encoder = OneHotEncoder(sparse_output=False)
  encoded_cols = encoder.fit_transform(df[columns])
  cat_feature_names = encoder.get_feature_names_out(columns)
  encoded_df = pd.DataFrame(encoded_cols, columns=cat_feature_names)
  encoded_df = encoded_df.reset_index(drop=True)
  df = df.reset_index(drop=True)
  df = pd.concat([df, encoded_df], axis=1)
  df = df.drop(columns=columns)
  return df, encoder, cat_feature_names

def calculate_ts_external_data(
    cfg: dict,
    external_df: dict[pd.DataFrame],
    end_date: pd.Timestamp | str,
    step_size: int = 1,
    window_size: int = 6
  ) -> dict:
  """
  Calculate client-independent (external) time-series features for a given window.

  Args:
    cfg (dict): The configuration dictionary.
    external_df (dict[pd.DataFrame]): The DataFrame containing the external data.
    end_date (pd.Timestamp | str): The end date of the window.
    step_size (int): The size of the step in months.
    window_size (int): The size of the window in months.

  Returns:
    dict: The time-series features dictionary.
  """

  external_df = external_df.copy()
  window_end = pd.date_range(end=end_date, periods=window_size, freq=f'{step_size}MS')
  window_start = window_end - pd.DateOffset(months=step_size)

  weather = {}
  cpi = {}
  gdp = {}
  unemployment = {}
  border_crossings = {}
  rent = {}

  def calculate_weather(df_weather: pd.DataFrame, year: int, month: int) -> dict:
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather = df_weather[df_weather['datetime'].dt.year == year]
    df_weather = df_weather[df_weather['datetime'].dt.month == month]

    _weather_feats = {}
    for feat in cfg['PREPROCESS']['EXTERNAL_FEATS']['WEATHER']:
      _weather_feats[feat] = df_weather[feat].mean()

    return _weather_feats
    
  def calculate_rent(df_rent: pd.DataFrame, year: int) -> float:
    df_rent = df_rent[df_rent['REF_DATE'].dt.year == year]
    rent = df_rent['VALUE'].mean()
    return rent
  
  def calculate_cpi(df_cpi: pd.DataFrame, year: int, month: int) -> float:
    df_cpi = df_cpi[df_cpi['REF_DATE'].dt.year == year]
    df_cpi = df_cpi[df_cpi['REF_DATE'].dt.month == month]
    cpi = df_cpi['VALUE'].mean()
    return cpi

  def calculate_gdp(df_gdp: pd.DataFrame, year: int, month: int) -> float:
    df_gdp = df_gdp[df_gdp['REF_DATE'].dt.year == year]
    df_gdp = df_gdp[df_gdp['REF_DATE'].dt.month == month]
    gdp = df_gdp['VALUE'].mean()
    return gdp
  
  def calculate_border_crossings(df_border_crossings: pd.DataFrame, year: int, month: int) -> float:
    df_border_crossings = df_border_crossings[df_border_crossings['Date'].dt.year == year]
    df_border_crossings = df_border_crossings[df_border_crossings['Date'].dt.month == month]
    border_crossings = df_border_crossings['Intake'].sum()
    return border_crossings
  
  for i, (start, end) in enumerate(zip(window_start, window_end)):
    # Weather
    year = start.year
    month = start.month
    weather_feats = calculate_weather(external_df['weather'], year, month)
    for key, value in weather_feats.items():
      weather[f'-{len(window_start) - 1 - i}_month_{key}'] = value

    # Rent
    rent_feats = calculate_rent(external_df['rent'], year)
    rent[f'-{len(window_start) - 1 - i}_month_rent'] = rent_feats

    # CPI
    cpi_feats = calculate_cpi(external_df['cpi'], year, month)
    cpi[f'-{len(window_start) - 1 - i}_month_cpi'] = cpi_feats

    # GDP
    gdp_feats = calculate_gdp(external_df['gdp'], year, month)
    gdp[f'-{len(window_start) - 1 - i}_month_gdp'] = gdp_feats

    # Border crossings
    border_crossings_feats = calculate_border_crossings(external_df['border_crossings'], year, month)
    border_crossings[f'-{len(window_start) - 1 - i}_month_border_crossings'] = border_crossings_feats

    # reorder the weather dictionary by key
    weather = dict(sorted(weather.items()))
  feats = {**weather, **rent, **cpi, **gdp, **unemployment, **border_crossings}
  return feats

def calculate_client_ts_feats(
    cfg: dict,
    feat_window_df: pd.DataFrame,
    external_df: dict[pd.DataFrame],
    end_date: pd.Timestamp | str,
    step_size: int = 1,
    window_size: int = 6
  ) -> pd.DataFrame:
  """
  Calculate client-dependent (HIFIS and external) time-series features for a given window.

  Args:
    cfg (dict): The configuration dictionary.
    feat_window_df (DataFrame): The DataFrame containing the features data.
    external_df (dict[DataFrame]): The DataFrame containing the external data.
    end_date (pd.Timestamp | str): The end date of the window.
    step_size (int): The size of the step in months.
    window_size (int): The size of the window in months.

  Returns:
    DataFrame: The time-series features DataFrame.
  """
  feat_window_df = feat_window_df.copy()
  
  window_end = pd.date_range(end=end_date, periods=window_size, freq=f'{step_size}MS')
  window_start = window_end - pd.DateOffset(months=step_size)

  def assemble_ts(client_df: pd.DataFrame) -> pd.Series:
    """
    Assemble the time-series features for a given client.

    Args:
      client_df (DataFrame): The DataFrame containing the client data.

    Returns:
      Series: The time-series features for the client.
    """
    
    stays = {}
    unemployment = {}

    def calculate_stays(df_window: pd.DataFrame) -> pd.Series:
      df_window['StayDuration'] = (df_window['StayDateEnd'] - df_window['StayDateStart']).dt.days
      return df_window['StayDuration'].sum()

    def calculate_unemployment(df_unemployment: pd.DataFrame, year: int, month: int, client_age: int) -> float:
      df_unemployment = df_unemployment[df_unemployment['REF_DATE'].dt.year == year]
      df_unemployment = df_unemployment[df_unemployment['REF_DATE'].dt.month == month]
      df_unemployment = df_unemployment[df_unemployment['AgeLB'] <= client_age]
      df_unemployment = df_unemployment[df_unemployment['AgeUB'] >= client_age]
      if df_unemployment.empty:
        unemployment = -1
      else:
        unemployment = df_unemployment['VALUE'].mean()
      return unemployment
    
    for i, (start, end) in enumerate(zip(window_start, window_end)):

      year = start.year
      month = start.month

      # Stays
      df_window = clip_df(client_df, start, end)
      stays[f'-{len(window_start) - 1 - i}_month_stays'] = calculate_stays(df_window)

      # Unemployment
      client_age = client_df['Age'].iloc[0]
      unemployment_feats = calculate_unemployment(external_df['unemployment'], year, month, client_age)
      unemployment[f'-{len(window_start) - 1 - i}_month_unemployment'] = unemployment_feats

    feats = pd.Series({**stays, **unemployment})
    
    return feats
  
  tqdm.pandas()
  ts_feats = feat_window_df.groupby('ClientHash', as_index=False).apply_parallel(assemble_ts, num_processes=cfg['PREPROCESS']['PARALLEL_COMPUTING']['N_PROCESS'])

  return ts_feats

def calculate_gt(gt_window_df: pd.DataFrame) -> pd.DataFrame:
  """
  Calculate the ground truth for a given window.

  Args:
    gt_window_df (DataFrame): The DataFrame containing the ground truth data.

  Returns:
    DataFrame: The ground truth DataFrame.
  """
  gt_window_df = gt_window_df.copy()
  gt_window_df['StayDuration'] = (gt_window_df['StayDateEnd'] - gt_window_df['StayDateStart']).dt.days
  gt = gt_window_df.groupby('ClientHash', as_index=False)['StayDuration'].sum()
  gt = gt.rename(columns={'StayDuration': 'GT'})
  
  return gt

def aggregate_features(cfg: dict, df: pd.DataFrame, groupby_cols: list[str], cat_feats: list[str], numerical_feats: list[str]) -> pd.DataFrame:
  """
  Aggregate features in a DataFrame.

  Args:
    cfg (dict): The configuration dictionary.
    df (DataFrame): The DataFrame to aggregate
    groupby_cols (list): The columns to group by.
    client_level_feats (list): The client-level features to aggregate by most recent occurance.
    visit_level_feats (list): The visit-level features to aggregate by summing.
    numerical_feats (list): The numerical features to aggregate by taking the mean.

  Returns:
    DataFrame: The aggregated DataFrame.
  """
  df = df.copy()
  cat_feats = sorted(cat_feats)
  client_level_feats = cfg['PREPROCESS']['AGGREGATION']['CLIENT_LEVEL_FEATS']
  visit_level_feats = cfg['PREPROCESS']['AGGREGATION']['VISIT_LEVEL_FEATS']
  agg_funcs = {col: 'mean' for col in numerical_feats}

  agg_funcs.update({col: lambda x: 1 if any(x) else 0 for col in cat_feats if col.split('_')[0] in visit_level_feats})
  # agg_funcs.update({'StayDuration': 'sum'})

  agg_funcs.update({col: lambda x: x.iloc[-1] for col in cat_feats if col.split('_')[0] in client_level_feats})
  agg_funcs.update({'Housed': lambda x: x.iloc[-1]})

  # progress apply the aggregation functions
  tqdm.pandas()
  df = df.groupby(groupby_cols, as_index=False).apply_parallel(lambda x: x.agg(agg_funcs), num_processes=cfg['PREPROCESS']['PARALLEL_COMPUTING']['N_PROCESS'])
  return df




def update_reason_for_discharge(df: pd.DataFrame, gt_start_date: pd.Timestamp | str) -> pd.DataFrame:
  """
  Update the ReasonForDischarge for the clients that have ongoing stays at the given date.

  Args:
    df (DataFrame): The DataFrame containing the data.
    gt_start_date (str): The start date of the ground truth window.

  Returns:
    DataFrame: The DataFrame with the ReasonForDischarge updated.
  """
  # get the index of last column that starts with "ReasonForDischarge_"
  last_rfd_col = [col for col in df.columns if col.startswith('ReasonForDischarge_')][-1]
  last_rfd_index = df.columns.get_loc(last_rfd_col)

  # insert a column for "ReasonForDischarge_Ongoing" at at the  set the values to 0
  df.insert(last_rfd_index + 1, 'ReasonForDischarge_Ongoing', 0.0)

  def update_client_rfd(df_client):
    df_client = df_client.sort_values(by='StayDateStart').reset_index(drop=True)
    last_record = df_client.iloc[-1]
    if last_record['StayDateEnd'] >= gt_start_date or pd.isnull(last_record['StayDateEnd']):
      # set the ReasonForDischarge_Ongoing to 1 if the client has an ongoing stay at the given date
      df_client.loc[df_client.index[-1], 'ReasonForDischarge_Ongoing'] = 1.0

      # set every other ReasonForDischarge to 0
      for col in df_client.columns:
        if col.startswith('ReasonForDischarge_') and col != 'ReasonForDischarge_Ongoing':
          df_client.loc[df_client.index[-1], col] = 0.0

    return df_client
  
  tqdm.pandas()
  return df.groupby('ClientHash').apply_parallel(update_client_rfd, num_processes=cfg['PREPROCESS']['PARALLEL_COMPUTING']['N_PROCESS']).reset_index(drop=True)


def preprocess_features_and_calculate_gt(
    cfg: dict,
    df_window: pd.DataFrame,
    external_data: dict[pd.DataFrame],
    cat_feats: list[str],
    numerical_feats: list[str],
    ts_window_size: int,
    ts_step_size: int,
    gt_start_date:pd.Timestamp | str,
    gt_duration: int = 6
  ) -> pd.DataFrame:
  """
  Preprocess the features and calculate the ground truth for a given window of data.

  Args:
    df_window (DataFrame): The DataFrame containing the data for the window.
    external_data (dict): A dictionary containing the external data.
    cat_feats (list): The categorical features to use.
    numerical_feats (list): The numerical features to use.
    ts_window_size (int): The size of the time-series window in months.
    ts_step_size (int): The step size of the time-series window in months.
    gt_start_date (str): The start date of the ground truth window.
    gt_duration (int): The duration of the ground truth window.

  Returns:
    DataFrame: The preprocessed data for the window.
  """

  df_gt_window = clip_df(df_window, gt_start_date, gt_start_date + pd.DateOffset(months=gt_duration))
  df_feat_window = remove_after_date(df_window, gt_start_date)


  log_time("AGGREGATING FEATURES", 'y')
  df_feat_window['StayDuration'] = (df_feat_window['StayDateEnd'] - df_feat_window['StayDateStart']).dt.days
  df_feats = aggregate_features(cfg, df_feat_window, ['ClientHash'], cat_feats, numerical_feats)
  
  log_time("CALCULATING TIME SERIES FEATURES", 'y')
  df_ts_feats = calculate_client_ts_feats(cfg, df_feat_window, external_data, gt_start_date, window_size=ts_window_size, step_size=ts_step_size)
  external_ts_feats = calculate_ts_external_data(cfg, external_data, gt_start_date, step_size=ts_step_size, window_size=ts_window_size)
  
  for key, value in external_ts_feats.items():
    df_ts_feats.loc[:, key] = value

  log_time("CALCULATING GROUND TRUTH", 'y')
  df_gt = calculate_gt(df_gt_window)

  log_time("MERGING DATA", 'y')
  df_window = pd.merge(df_feats, df_ts_feats, on='ClientHash', how='left')

  df_window = pd.merge(df_window, df_gt, on='ClientHash', how='left')
  df_window['GT'] = df_window['GT'].fillna(0)
  gt_bin = df_window[['-5_month_stays', '-4_month_stays', '-3_month_stays', '-2_month_stays', '-1_month_stays', '-0_month_stays', 'GT']].sum(axis=1) > 180
  df_window['GT_binary'] = np.where(gt_bin, 1, 0)
  df_window.insert(0, 'Date', gt_start_date)
  
  return df_window



def preprocess(cfg: dict) -> pd.DataFrame:
  """
  Preprocess the HIFIS dataset.

  Returns:
    DataFrame: The preprocessed data.
  """

  time_start = time.time()

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

  min_date = pd.to_datetime(cfg['PREPROCESS']['SLIDING_WINDOW']['START_DATE'])
  max_date = pd.to_datetime(main_data['StayDateStart'].max())

  log_time("PREPROCESSING FEATURES")
  GT_DURATION = cfg['PREPROCESS']['GT']['DURATION']
  TS_WINDOW_SIZE = cfg['PREPROCESS']['SLIDING_WINDOW']['WINDOW_SIZE']
  TS_STEP_SIZE = cfg['PREPROCESS']['SLIDING_WINDOW']['WINDOW_STEP']
  
  gt_start_dates = pd.date_range(min_date + pd.DateOffset(months=TS_WINDOW_SIZE) , max_date - pd.DateOffset(months=GT_DURATION), freq='MS')

  df_temp = pd.DataFrame()
  for i, gt_start_date in enumerate(gt_start_dates):
    log_time(f"PROCESSING WINDOW {i + 1}/{len(gt_start_dates)} - GT_START_DATE: {gt_start_date}", 'c')
    df_window = preprocess_features_and_calculate_gt(
      cfg,
      main_data,
      external_data=external_data,
      cat_feats=[*cat_feature_names, 'ReasonForDischarge_Ongoing'],
      numerical_feats=['Age'],
      ts_window_size=TS_WINDOW_SIZE,
      ts_step_size=TS_STEP_SIZE,
      gt_start_date=gt_start_date,
      gt_duration=GT_DURATION
    )
    df_temp = pd.concat([df_temp, df_window], axis=0)

  log_time("SAVING DATA", 'c')
  df_temp.to_csv(cfg['PATHS']['DATA']['PROCESSED'], index=False)

  time_end = time.time()
  print(f"{Fore.GREEN}Process completed in:{Style.RESET_ALL} {time.strftime('%H:%M:%S', time.gmtime(time_end - time_start))}")

  

if __name__ == '__main__':
  cfg = load_config('config.yml')
  print(f"{Fore.GREEN}PERFORMING PREPROCESSING IN PARALLEL: {cfg['PREPROCESS']['PARALLEL_COMPUTING']['N_PROCESS']} CPU CORES{Style.RESET_ALL} ")
  preprocess(cfg)
  




  