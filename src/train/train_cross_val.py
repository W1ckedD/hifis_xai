import os
import yaml
import numpy as np
import lightning as pl
from src.utils import load_config
from src.data.nested_kfold import NestedKFold
from src.model.model import ModelRegression
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint




def main():
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
  
  n_iters = cfg['TRAIN']['NESTED_KFOLD']['N_ITERS']

  batch_size = cfg['TRAIN']['BATCH_SIZE']
  num_workers = cfg['TRAIN']['NUM_WORKERS']
  lr = cfg['TRAIN']['LEARNING_RATE']
  epochs = cfg['TRAIN']['EPOCHS']
  patience = cfg['TRAIN']['EARLY_STOPPING_PATIENCE']
  ckpt_dir = cfg['PATHS']['MODEL']['CHECKPOINTS']

  chronic_threshold = cfg['TRAIN']['MODEL']['CHRONIC_THRESHOLD']

  DATA_DIR = cfg['PATHS']['DATA']['PROCESSED']
  

  test_acc = {}
  test_pre = {}
  test_rec = {}
  test_f1 = {}
  test_mae = {}


  for i in range(n_iters):
    import wandb
    from lightning.pytorch.loggers import WandbLogger
    
    dm = NestedKFold(
      data_dir=DATA_DIR,
      batch_size=batch_size,
      n_iters=n_iters,
      ts_seq_len=ts_seq_len,
      num_workers=num_workers
    )
    dm.current_iter = i
    dm.prepare_data()
    dm.load_next_iter()
    print(f"Data location: {dm.data_dir}")
    logger = WandbLogger(project=PROJECT_NAME, name=f"{EXP}_iter_{dm.current_iter}")
    
    callbacks = [
      ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', dirpath=ckpt_dir, filename=f'{EXP}_iter_{dm.current_iter}_' + '{epoch}_{val_loss}'),
      EarlyStopping(monitor='val_loss', patience=patience, mode='min'),
    ]
    
    trainer = pl.Trainer(
      logger=logger,
      callbacks=callbacks,
      max_epochs=epochs,
      accelerator=ACCELERATOR,
      strategy=STRATEGY,
    )

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
      chronic_threshold=chronic_threshold,
    )

    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)

    test_acc[i] = trainer.callback_metrics['test_accuracy'].cpu().numpy()
    test_pre[i] = trainer.callback_metrics['test_precision'].cpu().numpy()
    test_rec[i] = trainer.callback_metrics['test_recall'].cpu().numpy()
    test_f1[i] = trainer.callback_metrics['test_f1'].cpu().numpy()
    test_mae[i] = trainer.callback_metrics['test_mae'].cpu().numpy()

    logger.experiment.finish()
    model = None
    trainer = None
    logger = None
    callbacks = None


  print(f"Test Accuracy: {np.mean(list(test_acc.values()))}")
  print(f"Test Precision: {np.mean(list(test_pre.values()))}")
  print(f"Test Recall: {np.mean(list(test_rec.values()))}")
  print(f"Test F1: {np.mean(list(test_f1.values()))}")
  print(f"Test MAE: {np.mean(list(test_mae.values()))}")

  run = wandb.init(project=PROJECT_NAME, name=f"{EXP}_avg")
  run.log({'AVG Metrics': wandb.Table(columns=['Accuracy', 'Precision', 'Recall', 'F1', 'MAE'], data=[[np.mean(list(test_acc.values())), np.mean(list(test_pre.values())), np.mean(list(test_rec.values())), np.mean(list(test_f1.values())), np.mean(list(test_mae.values()))]])})

  run.finish()


if __name__ == '__main__':
  main()