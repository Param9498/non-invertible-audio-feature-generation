import os
import random

import torch
from torch.utils.data import Dataset
import torchaudio.functional as F

import numpy as np
import pandas as pd

import tqdm


def binarySearch(data, val):
    highIndex = len(data)-1
    lowIndex = 0
    while highIndex > lowIndex:
            index = (highIndex + lowIndex) // 2
            sub = data[index]
            if data[lowIndex] == val:
                    return [lowIndex, lowIndex]
            elif sub == val:
                    return [index, index]
            elif data[highIndex] == val:
                    return [highIndex, highIndex]
            elif sub > val:
                    if highIndex == index:
                            return sorted([highIndex, lowIndex])
                    highIndex = index
            else:
                    if lowIndex == index:
                            return sorted([highIndex, lowIndex])
                    lowIndex = index
    return sorted([highIndex, lowIndex])


class AudioDataset(Dataset):

    def __init__(self, root_dir, transform=None, num_audios = -1, num_frames = -1, return_amp = True, emb_means = None, emb_stds = None, spec_means = None, spec_stds = None):
        
        self.root_dir = root_dir
        self.embeddings_dir = os.path.join(self.root_dir, 'embeddings_6144')
        self.spectrograms_dir = os.path.join(self.root_dir, 'spectrograms')
        self.audio_pred_dir = os.path.join(self.root_dir, 'audio_prep')
        self.transform = transform
        self.num_audios = num_audios
        self.return_amp = return_amp
        self.emb_means = emb_means
        self.emb_stds = emb_stds
        self.spec_means = spec_means
        self.spec_stds = spec_stds
        self.num_frames = num_frames
        
        self.df = pd.read_csv(os.path.join(root_dir, 'number_of_frames_per_audio.csv'))
        if num_audios > 0 and isinstance(num_audios, int):
            self.df = self.df.head(num_audios)
        self.cumulative_sum = self.df['number_of_frames'].cumsum()
        
                
    def __len__(self):
        if self.num_frames != -1:
            return self.num_frames
        else:
            return self.df['number_of_frames'].sum()

    def __getitem__(self, idx):
        
        if self.num_frames != -1:
            idx = idx % self.num_frames
        
        low_index, high_index = binarySearch(self.cumulative_sum, idx+1)
        file_name = self.df.iloc[high_index]['file_name']
        emb_path = os.path.join(self.embeddings_dir, file_name)        
        spec_path = os.path.join(self.spectrograms_dir, file_name)
        audio_prep_path = os.path.join(self.audio_pred_dir, file_name)
        
        if low_index == 0 and high_index == 0:
            frame_idx = idx
        else:
            frame_idx = idx - self.cumulative_sum[low_index]
            
        with open(emb_path, 'rb') as f:
            emb = np.load(f)
        with open(spec_path, 'rb') as f:
            spec = np.load(f)
        with open(audio_prep_path, 'rb') as f:
            audio_prep = np.load(f)
        
        emb_tensor = torch.from_numpy(emb[frame_idx])
        spec_tensor = torch.from_numpy(spec[frame_idx]).permute(2, 0, 1)
        audio_prep_tensor = torch.from_numpy(audio_prep[frame_idx])
        
        if self.emb_means is not None and self.emb_stds is not None:
            emb_tensor = ( emb_tensor - torch.tensor(self.emb_means) ) / torch.tensor(self.emb_stds)
        
                
        if self.return_amp is True:
            spec_tensor_amp = F.DB_to_amplitude(x = spec_tensor, ref = 1, power = 0.5)
            
            if self.spec_means is not None and self.spec_stds is not None:
                spec_tensor_amp = ( spec_tensor_amp - torch.tensor(self.spec_means) ) / torch.tensor(self.spec_stds)
                spec_tensor_amp = spec_tensor_amp.float()
            
            return emb_tensor, spec_tensor_amp, audio_prep_tensor, file_name, torch.tensor(frame_idx)
        else:
            return emb_tensor, spec_tensor, audio_prep_tensor, file_name, torch.tensor(frame_idx)

