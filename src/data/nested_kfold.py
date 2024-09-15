import os
import torch
import pandas as pd
import numpy as np
import lightning as L
from sklearn.preprocessing import StandardScaler
from joblib import dump
from torch.utils.data import DataLoader, TensorDataset
from src.utils import load_config

class NestedKFold(L.LightningDataModule):
  def __init__(
    self,
    data_dir: str,
    n_iters: int,
    ts_seq_len: int = 6,
    batch_size: int = 1024,
    num_workers: int = 4
  ) -> None:
    """
      Constructor for NestedKFold class.
      This is the PyTorch implementation for the Nested K-Fold method introduced in https://github.com/aildnont/HIFIS-model.
      
      At each iteration, the data related to the most recent time step is used for testing, the data related to the second most recent time step is used for validation, and the rest of the data is used for training. When moving to the next iteration, the most recent time step is dropped and the folds are updated using the same logic but applied to the updated dataset. For more detailed information, please refer to the original paper.

      Args:
        data_dir: str: Path to the data file
        n_iters: int: Number of iterations the experiments will be run
        batch_size: int: Batch size for the data loaders
        num_workers: int: Number of workers for the data loaders
    """

    super().__init__()
    self.data_dir = data_dir
    self.n_iters = n_iters
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.ts_seq_len = ts_seq_len

    self.current_iter = 0
    self.cfg = load_config('config.yml')

  def prepare_data(self) -> None:
    """
    This method is used to load the data and prepare it for the Nested K-Fold method.
    Loads the data, calculates the total number of time series features, and the total number of static features.
    Sets them as properties to the object for future use.
    Calculates the sum_stays feature which is the sum of number of stays during the most recent 6 months.
    """
    # Load data
    self.data = pd.read_csv(self.data_dir)
    print(f"Total records: {self.data.shape[0]}")

    ts_feat_names = [feat for feat in self.data.columns if '_month_' in feat]
    ts_stay_names = ['-5_month_stays', '-4_month_stays', '-3_month_stays', '-2_month_stays', '-1_month_stays', '-0_month_stays']
    self.data['sum_stays'] = self.data[ts_stay_names].sum(axis=1)
    self.x_static_input_dim = len(self.data.columns) - len(ts_feat_names) - 6 # 6 is for GT, GT_binary, Date, ClientHash, sum_stays, ReasonForDischarge_Ongoing
    self.x_ts_input_dim = len(ts_feat_names) // self.ts_seq_len

  def load_next_iter(self) -> None:
    """
    This method is used to load the next iteration of the Nested K-Fold method.
    It splits the data into train, validation, and test sets.
    Scales the features using StandardScaler.
    Dumps the scaler object to a file for future use.
    """

    assert self.current_iter < self.n_iters, "All iterations have been loaded"
  
    print(f"Loading fold {self.current_iter + 1} / {self.n_iters}")

    unique_dates = self.data['Date'].unique()
    if self.current_iter != 0:
      unique_dates = unique_dates[:-self.current_iter]

    # Split data into train and test

    # Set the last date as the test date
    test_dates = unique_dates[-1:]
    val_dates = unique_dates[-2:-1]
    train_dates = unique_dates[:-2]

    print(f"Train dates: {train_dates}")
    print(f"Val dates: {val_dates}")
    print(f"Test dates: {test_dates}")

    train_data = self.data[self.data['Date'].isin(train_dates)]
    val_data = self.data[self.data['Date'].isin(val_dates)]
    test_data = self.data[self.data['Date'].isin(test_dates)]

    train_data = train_data.drop(columns=['Date', 'ClientHash', 'ReasonForDischarge_Ongoing'])

    val_data = val_data.drop(columns=['Date', 'ClientHash', 'ReasonForDischarge_Ongoing'])
    test_data = test_data.drop(columns=['Date', 'ClientHash', 'ReasonForDischarge_Ongoing'])

    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    

    scaler = StandardScaler()
    skip_cols = ['GT', 'GT_binary', 'sum_stays']
    scale_cols = train_data.columns.difference(skip_cols)

    train_data[scale_cols] = scaler.fit_transform(train_data[scale_cols])
    val_data[scale_cols] = scaler.transform(val_data[scale_cols])
    test_data[scale_cols] = scaler.transform(test_data[scale_cols])

    os.makedirs(self.cfg["PATHS"]["COMPONENTS"]["SCALER"], exist_ok=True)
    dump(scaler, f'{self.cfg["PATHS"]["COMPONENTS"]["SCALER"]}{self.current_iter}.pkl')

    self.train_data = train_data
    self.val_data = val_data
    self.test_data = test_data

    self.current_iter += 1

  def create_feature_tensors(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates static and time series feature tensors from the input dataframe values

    Args:
      df: pd.DataFrame: Input dataframe

    Returns:
      tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing static features, time series features, ground truth, and sum_stays
      
    """
    gt = df['GT']

    ts_feat_names = [feat for feat in df.columns if '_month_' in feat]
    ts_feat_names = sorted(ts_feat_names, key=lambda x: int(x.split('_month_')[0]))
    
    ts_feats = df[ts_feat_names]
    sum_stays = df['sum_stays']

    static_feats = df.drop(columns=['GT', 'GT_binary', 'sum_stays', *ts_feat_names])

    static_feats = torch.tensor(static_feats.values, dtype=torch.float32)
    ts_feats = torch.tensor(ts_feats.values, dtype=torch.float32)
    gt = torch.tensor(gt.values, dtype=torch.float32)
    sum_stays = torch.tensor(sum_stays.values, dtype=torch.float32)

    return static_feats, ts_feats, gt, sum_stays

  def setup(self, stage: str | None = None) -> None:
    """
    Creates the PyTorch datasets for training, validation, and testing based on different stages of the training procedure.
    """
    if stage == 'fit' or stage is None:
      self.train_static_feats, self.train_ts_feats, self.train_gt, self.train_total_stays = self.create_feature_tensors(self.train_data)
      self.val_static_feats, self.val_ts_feats, self.val_gt, self.val_total_stays = self.create_feature_tensors(self.val_data)
      
      self.train_dataset = TensorDataset(self.train_static_feats, self.train_ts_feats, self.train_gt, self.train_total_stays)
      self.val_dataset = TensorDataset(self.val_static_feats, self.val_ts_feats, self.val_gt, self.val_total_stays)
    if stage == 'test' or stage is None:
      self.test_static_feats, self.test_ts_feats, self.test_gt, self.test_total_stays = self.create_feature_tensors(self.test_data)
      self.test_dataset = TensorDataset(self.test_static_feats, self.test_ts_feats, self.test_gt, self.test_total_stays)

  def train_dataloader(self) -> DataLoader:
    return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
  
  def val_dataloader(self) -> DataLoader:
    return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
  
  def test_dataloader(self) -> DataLoader:
    return DataLoader(self.test_dataset, batch_size=10000, num_workers=self.num_workers, shuffle=False)
  