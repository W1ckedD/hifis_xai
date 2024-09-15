import os
import math
import torch
import wandb
import numpy as np
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from src.utils import load_config
from src.data.nested_kfold import NestedKFold
from src.model.model import ModelRegression


def main() -> None:
  pl.seed_everything(42)

  cfg = load_config('config.yml')
  PROJECT_NAME = cfg['TRAIN']['WANDB']['PROJECT_NAME']
  EXP = cfg['TRAIN']['WANDB']['EXP']
  ACCELERATOR = cfg['TRAIN']['ACCELERATOR']
  STRATEGY = cfg['TRAIN']['STRATEGY']

  hidden_dim = cfg['TRAIN']['MODEL']['MLP_HIDDEN_DIM']
  n_hidden = cfg['TRAIN']['MODEL']['MLP_N_HIDDEN']
  output_dim = 1
  dropout_p = cfg['TRAIN']['MODEL']['DROPOUT_PROB']

  lstm_hidden_dim = cfg['TRAIN']['MODEL']['LSTM_HIDDEN_DIM']
  n_lstm_hidden = cfg['TRAIN']['MODEL']['LSTM_N_HIDDEN']
  n_ts_feat = cfg['TRAIN']['MODEL']['NUM_TS_FEATS']
  ts_seq_len = cfg['TRAIN']['MODEL']['TS_SEQ_LEN']

  batch_size = cfg['TRAIN']['BATCH_SIZE']
  num_workers = cfg['TRAIN']['NUM_WORKERS']
  lr = cfg['TRAIN']['LEARNING_RATE']
  epochs = cfg['TRAIN']['EPOCHS']
  patience = cfg['TRAIN']['EARLY_STOPPING_PATIENCE']
  ckpt_dir = cfg['PATHS']['MODEL']['CHECKPOINTS']

  chronic_threshold = cfg['TRAIN']['MODEL']['CHRONIC_THRESHOLD']

  DATA_DIR = cfg['PATHS']['DATA']['PROCESSED']

  # Initialize the logger
  logger = WandbLogger(
    name=f"{EXP}_single_train",
    project=PROJECT_NAME
  )

  # Initialize the data module
  dm = NestedKFold(
    data_dir=DATA_DIR,
    n_iters=1,
    ts_seq_len=ts_seq_len,
    batch_size=batch_size,
    num_workers=num_workers
  )

  # Initialize the callbacks
  callbacks = [
    ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', dirpath=ckpt_dir, filename=f'{EXP}_single_train' + '-{epoch}-{val_loss:.2f}'),
    EarlyStopping(monitor='val_loss', patience=patience, mode='min'),
  ]
  # Initialize the trainer
  trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    max_epochs=epochs,
    accelerator=ACCELERATOR,
    strategy=STRATEGY,
  )

  dm.prepare_data()
  dm.load_next_iter()
  print(f"Data location: {dm.data_dir}")

  # Initialize the model
  model = ModelRegression(
    input_dim=dm.x_static_input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    n_hidden=n_hidden,
    dropout_p=dropout_p,
    n_ts_feat=n_ts_feat,
    seq_len=ts_seq_len,
    n_lstm_hidden=n_lstm_hidden,
    lstm_hidden_dim=lstm_hidden_dim,
    lr=lr,
    chronic_threshold=chronic_threshold
  )

  # Train the model
  trainer.fit(model, dm)
  # Test the model
  trainer.test(model=model, datamodule=dm, ckpt_path='best')

  # Log test metrics
  test_acc = trainer.callback_metrics['test_accuracy'].item()
  test_pre = trainer.callback_metrics['test_precision'].item()
  test_rec = trainer.callback_metrics['test_recall'].item()
  test_f1 = trainer.callback_metrics['test_f1'].item()
  test_mae = trainer.callback_metrics['test_mae'].item()

  metrics_table = wandb.Table(
    columns=['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test MAE'],
    data=[[test_acc, test_pre, test_rec, test_f1, test_mae]]
  )

  logger.experiment.log({'Test Metrics': metrics_table})

  logger.experiment.finish()

if __name__ == '__main__':
  main()



