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

class AbstractModel(pl.LightningModule):
    
    def __init__(self, 
                 hparams, 
                 data_paths, 
                 dataset_model,
                 model, 
                 criterion = None):
        
        super().__init__()
        self.hparams = hparams
        
        self.data_paths = data_paths
        self.model = model
        
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion

        self.best_validation_loss = 1e6
        
        self.dataset_model = dataset_model
        
        self.best_validation_loss = 1e6
        
        if not hasattr(self.hparams, 'num_frames'):
            self.hparams.num_frames = -1
        if not hasattr(self.hparams, 'train_num_audios'):
            self.hparams.train_num_audios = -1
        if not hasattr(self.hparams, 'val_num_audios'):
            self.hparams.val_num_audios = -1
        if not hasattr(self.hparams, 'test_num_audios'):
            self.hparams.test_num_audios = -1
        if not hasattr(self.hparams, 'emb_means'):
            self.hparams.emb_means = None
        if not hasattr(self.hparams, 'emb_stds'):
            self.hparams.emb_stds = None
        if not hasattr(self.hparams, 'spec_means'):
            self.hparams.spec_means = None
        if not hasattr(self.hparams, 'spec_stds'):
            self.hparams.spec_stds = None
            
        self.list_of_files = []
#         self.list_of_frames = []
    
    def prepare_data(self):
        self.train_dataset = self.dataset_model(
            root_dir=self.data_paths['train'], 
            num_audios = self.hparams.train_num_audios, 
            num_frames = self.hparams.num_frames,
            return_amp = self.hparams.return_amp, 
            emb_means=self.hparams.emb_means, 
            emb_stds=self.hparams.emb_stds,
            spec_means = self.hparams.spec_means,
            spec_stds=self.hparams.spec_stds
        )
        self.val_dataset = self.dataset_model(
            root_dir=self.data_paths['val'], 
            num_audios = self.hparams.val_num_audios, 
            num_frames = self.hparams.num_frames,
            return_amp = self.hparams.return_amp, 
            emb_means=self.hparams.emb_means, 
            emb_stds=self.hparams.emb_stds,
            spec_means = self.hparams.spec_means,
            spec_stds=self.hparams.spec_stds
        )
        self.test_dataset = self.dataset_model(
            root_dir=self.data_paths['test'], 
            num_audios = self.hparams.test_num_audios, 
            num_frames = self.hparams.num_frames,
            return_amp = self.hparams.return_amp, 
            emb_means=self.hparams.emb_means, 
            emb_stds=self.hparams.emb_stds,
            spec_means = self.hparams.spec_means,
            spec_stds=self.hparams.spec_stds
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def _fix_dp_return_type(self, result, device):
        if isinstance(result, torch.Tensor):
            return result.to(device)
        if isinstance(result, dict):
            return {k:self._fix_dp_return_type(v, device) for k,v in result.items()}
        # Must be a number then
        return torch.Tensor([result]).to(device)

    def training_step(self, batch, batch_nb):
        x, y, audio_prep, file_name, i = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        if file_name not in self.list_of_files:
            self.list_of_files.append(file_name)
#             self.list_of_frames.append(i.item())
        result = self._fix_dp_return_type(result={'loss': loss, 'log':{'batch_train_loss': loss.detach().item()}}, device=self.device)
        return result

    def training_epoch_end(self, outputs):
        # OPTIONAL
        if self.trainer.num_gpus == 0 or self.trainer.num_gpus == 1:
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        else:
            avg_loss = 0
            i = 0
            for dataloader_outputs in outputs:
                for output in dataloader_outputs['loss']:
                    avg_loss += output
                    i += 1

            avg_loss /= i
        tensorboard_logs = {'train_loss': avg_loss}
        return {'train_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y, audio_prep, file_name, i = batch
        y_hat = self(x)
        
        return {'val_loss': F.mse_loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        if self.trainer.num_gpus == 0 or self.trainer.num_gpus == 1:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        else:
            avg_loss = 0
            i = 0
            for dataloader_outputs in outputs:
                for output in dataloader_outputs['val_loss']:
                    avg_loss += output
                    i += 1

            avg_loss /= i
        
        if avg_loss.item() < self.best_validation_loss:
            self.best_validation_loss = avg_loss.item()
            
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y, audio_prep, file_name, i = batch
        y_hat = self(x)
        
        return {'test_loss': F.mse_loss(y_hat, y)}

    def test_epoch_end(self, outputs):
        if self.trainer.num_gpus == 0 or self.trainer.num_gpus == 1:
            avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        else:
            avg_loss = 0
            i = 0
            for dataloader_outputs in outputs:
                for output in dataloader_outputs['test_loss']:
                    avg_loss += output
                    i += 1

            avg_loss /= i
        
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        if self.hparams.lr_type == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.lr_type == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
            
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_epoch, self.hparams.scheduler_step_size)
        return [optimizer], [scheduler]
