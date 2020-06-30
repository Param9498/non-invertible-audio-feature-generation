import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np

import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.core.saving import load_hparams_from_yaml

import os

from data.audio_dataset_v1 import AudioDataset
from models.placeholder_model import PlaceHolder

class AbstractModel(pl.LightningModule):
    
    def __init__(self, 
                 hparams, 
                 data_paths):
        
        super().__init__()
        self.hparams = hparams
        
        self.data_paths = data_paths
        
        self.model = PlaceHolder()
        
        self.criterion = nn.MSELoss()
        self.best_validation_loss = 1e6
    
    def prepare_data(self):
        self.train_dataset = AudioDataset(root_dir=self.data_paths['train'])
        self.val_dataset = AudioDataset(root_dir=self.data_paths['val'])
        self.test_dataset = AudioDataset(root_dir=self.data_paths['test'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y, i = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['batch_loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss}
        return {'train_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y, i = batch
        y_hat = self(x)
        
        return {'val_loss': F.mse_loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        if avg_loss.item() < self.best_validation_loss:
            self.best_validation_loss = avg_loss.item()
            
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        
        return {'test_loss': F.mse_loss(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_epoch, self.hparams.scheduler_step_size)
        return [optimizer], [scheduler]
