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

    def __init__(self, root_dir, transform=None, num_audios = -1, return_amp = True):
        
        self.root_dir = root_dir
        self.embeddings_dir = os.path.join(self.root_dir, 'embeddings_6144')
        self.spectrograms_dir = os.path.join(self.root_dir, 'spectrograms')
        self.transform = transform
        self.num_audios = num_audios
        self.return_amp = return_amp
        
        self.df = pd.read_csv(os.path.join(root_dir, 'number_of_frames_per_audio.csv'))
        if num_audios > 0 and isinstance(num_audios, int):
            self.df = self.df.head(num_audios)
        self.cumulative_sum = self.df['number_of_frames'].cumsum()
                
    def __len__(self):
        return self.df['number_of_frames'].sum()

    def __getitem__(self, idx):
        
        low_index, high_index = binarySearch(self.cumulative_sum, idx+1)
        file_name = self.df.iloc[high_index]['file_name']
        emb_path = os.path.join(self.embeddings_dir, file_name)        
        spec_path = os.path.join(self.spectrograms_dir, file_name)
        
        if low_index == 0 and high_index == 0:
            frame_idx = idx
        else:
            frame_idx = idx - self.cumulative_sum[low_index]
        
        with open(emb_path, 'rb') as f:
            emb = np.load(f)
        with open(spec_path, 'rb') as f:
            spec = np.load(f)
        
        emb_tensor = torch.from_numpy(emb[frame_idx])
        spec_tensor = torch.from_numpy(spec[frame_idx]).permute(2, 0, 1)
                
        if self.return_amp is True:
            spec_tensor_amp = F.DB_to_amplitude(x = spec_tensor, ref = 1, power = 0.5)
            return emb_tensor, spec_tensor_amp, torch.tensor(frame_idx)
        
        else:
            return emb_tensor, spec_tensor, torch.tensor(frame_idx)

