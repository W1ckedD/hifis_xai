import torch
import torch.nn as nn
import lightning as pl
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, PrecisionRecallCurve
from torchmetrics.regression import MeanAbsoluteError
import wandb

class NetBlock(nn.Module):
  def __init__(self, input_dim: int, output_dim: int, dropout_p: float = 0.44) -> None:
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.dropout_p = dropout_p

    self.block = nn.Sequential(
      nn.Linear(self.input_dim, self.output_dim),
      nn.ReLU(),
      nn.Dropout(self.dropout_p)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.block(x)


class Net(nn.Module):
  def __init__(
      self,
      input_dim: int,
      hidden_dim: int,
      output_dim: int = 1,
      n_hidden: int = 6,
      dropout_p: float = 0.44,
      n_ts_feat: int = 12,
      seq_len: int = 6,
      n_lstm_hidden: int = 1,
      lstm_hidden_dim: int = 4
    ) -> None:
    """
    Constructor for the Net class
    
    Args:
      - input_dim: int: Dimension of the static input features
      - hidden_dim: int: Dimension of the hidden MLP layers
      - output_dim: int: Dimension of the output layer
      - n_hidden: int: Number of hidden MLP layers
      - dropout_p: float: Dropout probability
      - n_ts_feat: int: Number of time series features
      - seq_len: int: Length of the sequence
      - n_lstm_hidden: int: Number of hidden layers in the LSTM
      - lstm_hidden_dim: int: Dimension of the hidden layers in the LSTM
    """
    super().__init__()
    self.n_ts_feat = n_ts_feat
    self.seq_len = seq_len
    self.n_lstm_hidden = n_lstm_hidden
    self.lstm_hidden_dim = lstm_hidden_dim

    self.input_dim = input_dim + self.seq_len * self.lstm_hidden_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.n_hidden = n_hidden
    self.layers = []
    self.layers.append(NetBlock(self.input_dim, self.hidden_dim * 2, dropout_p=dropout_p))
    self.layers.append(NetBlock(self.hidden_dim * 2, self.hidden_dim, dropout_p=dropout_p))
    for _ in range(self.n_hidden - 1): # The first layer is already added so we add n_hidden - 1 layers
      self.layers.append(NetBlock(self.hidden_dim, self.hidden_dim, dropout_p=dropout_p))

    self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    self.net = nn.Sequential(*self.layers)

    self.lstm = nn.LSTM(input_size=self.n_ts_feat, hidden_size=self.lstm_hidden_dim, num_layers=self.n_lstm_hidden, batch_first=True)
    self.tanh = nn.Tanh()
    

  def forward(self, static_feats: torch.Tensor, ts_feats: torch.Tensor) -> torch.Tensor:
    ts_feats = ts_feats.view(-1, self.seq_len, self.n_ts_feat) # Reshape the time series features
    lstm_out, _ = self.lstm(ts_feats) # Pass the time series features through the LSTM
    lstm_out = self.tanh(lstm_out) # Apply the tanh activation function
    lstm_out = lstm_out.flatten(start_dim=1) # Flatten the output of the LSTM
    x = torch.cat([static_feats, lstm_out], dim=1) # Concatenate the static features with the output of the LSTM

    return self.net(x) # Pass the concatenated features through the MLP


class ModelRegression(pl.LightningModule):
  def __init__(
      self,
      input_dim: int,
      hidden_dim: int,
      output_dim: int = 1,
      n_hidden: int = 6,
      dropout_p: float = 0.44,
      n_ts_feat: int = 12,
      seq_len: int = 6,
      n_lstm_hidden: int = 1,
      lstm_hidden_dim: int = 4,
      lr: float = 1e-3,
      chronic_threshold: float = 120.0
    ):
    super().__init__()
    self.automatic_optimization = False
    
    self.lr = lr
    
    self.chronic_threshold = chronic_threshold

    self.model = Net(
      input_dim=input_dim,
      hidden_dim=hidden_dim,
      output_dim=output_dim,
      n_hidden=n_hidden,
      dropout_p=dropout_p,
      n_ts_feat=n_ts_feat,
      seq_len=seq_len,
      n_lstm_hidden=n_lstm_hidden,
      lstm_hidden_dim=lstm_hidden_dim
    )
    # Print the model archictecture
    print(self.model)

    # Loss function
    self.criterion = nn.MSELoss()

    # Metrics
    self.accuracy = BinaryAccuracy()
    self.precision = BinaryPrecision()
    self.recall = BinaryRecall()
    self.f1 = BinaryF1Score()

    self.mae = MeanAbsoluteError()

    self.pr_curve = PrecisionRecallCurve(task='binary')

    # accumulate test results for PR curve
    self.test_outputs = []
    self.test_gts = []
    self.test_errors = []

  def forward(self, static_feats: torch.Tensor, ts_feats: torch.Tensor) -> torch.Tensor:
    return self.model(static_feats, ts_feats)
  
  def training_step(self, batch, batch_idx):
    self.model.train() # Set the model to training mode (activate dropout ...)
    optim = self.optimizers()
    optim.zero_grad()
    static_feats, ts_feats, gt, sum_stays = batch
    output = self.model(static_feats, ts_feats)

    loss = self.criterion(output.flatten(), gt.flatten())
    self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
    self.manual_backward(loss)

    optim.step()
    
  
  def validation_step(self, batch, batch_idx):
    self.model.eval()
    with torch.no_grad():
      static_feats, ts_feats, gt, sum_stays = batch
      output = self.model(static_feats, ts_feats)
      loss = self.criterion(output.flatten(), gt.flatten())

      chronic_threshold = self.chronic_threshold
      # Element wise sum of gt and sum_stays
      gt_binary = (gt + sum_stays > 182).float()
      # Element wise sum of output and sum_stays
      pred_binary = (output + sum_stays.view(-1, 1) > chronic_threshold).float()
      
      acc = self.accuracy(pred_binary, gt_binary.view(-1, 1))
      prec = self.precision(pred_binary, gt_binary.view(-1, 1))
      rec = self.recall(pred_binary, gt_binary.view(-1, 1))
      f1 = self.f1(pred_binary, gt_binary.view(-1, 1))

      self.log('val_accuracy', acc, prog_bar=True, on_step=False, on_epoch=True)
      self.log('val_precision', prec, prog_bar=True, on_step=False, on_epoch=True)
      self.log('val_recall', rec, prog_bar=True, on_step=False, on_epoch=True)
      self.log('val_f1', f1, prog_bar=True, on_step=False, on_epoch=True)

      mae = self.mae(output.flatten(), gt.flatten())
      self.log('val_mae', mae, prog_bar=True, on_step=False, on_epoch=True)
      self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

  def test_step(self, batch, batch_idx):
    self.model.eval()
    with torch.no_grad():
      static_feats, ts_feats, gt, sum_stays = batch
      output = self.model(static_feats, ts_feats)
      loss = self.criterion(output.flatten(), gt.flatten())
      
      chronic_threshold = self.chronic_threshold
      gt_binary = (gt + sum_stays > 182).float()
      pred_binary = (output + sum_stays.view(-1, 1) > chronic_threshold).float()
      
      acc = self.accuracy(pred_binary, gt_binary.view(-1, 1))
      prec = self.precision(pred_binary, gt_binary.view(-1, 1))
      rec = self.recall(pred_binary, gt_binary.view(-1, 1))
      f1 = self.f1(pred_binary, gt_binary.view(-1, 1))

      mae = self.mae(output.flatten(), gt.flatten())
      self.log('test_mae', mae, prog_bar=True, on_step=False, on_epoch=True)

      self.log('test_accuracy', acc, prog_bar=True, on_step=False, on_epoch=True)
      self.log('test_precision', prec, prog_bar=True, on_step=False, on_epoch=True)
      self.log('test_recall', rec, prog_bar=True, on_step=False, on_epoch=True)
      self.log('test_f1', f1, prog_bar=True, on_step=False, on_epoch=True)

      self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

      self.test_outputs.append((output.flatten() + sum_stays) / 182)
      self.test_gts.append(gt_binary.int())
      
      self.test_errors.append(output.flatten() - gt.flatten())

      


  def on_test_epoch_end(self):
    self.plot_pr_curve(torch.cat(self.test_outputs), torch.cat(self.test_gts))
    self.plot_test_error_distribution()

    self.test_outputs = []
    self.test_gts = []
    self.test_errors = []

  def plot_test_error_distribution(self):
    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.hist(torch.cat(self.test_errors).cpu().numpy(), bins=184)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution For All GT')

    self.logger.experiment.log({'error_distribution': wandb.Image(plt)})


  def plot_pr_curve(self, output, gt):
    plt.clf()
    _pre, _rec, thresholds = self.pr_curve(output, gt)
    plt.figure(figsize=(10, 10))
    plt.plot(_rec.cpu().numpy(), _pre.cpu().numpy())
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    self.logger.experiment.log({'pr_curve': wandb.Image(plt)})

  def configure_optimizers(self):
    return torch.optim.Adam(self.model.parameters(), lr=self.lr)
  
