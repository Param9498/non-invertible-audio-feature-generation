import os
import random

import torch
from torch.utils.data import Dataset

import numpy as np


class AudioDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        
        self.list_of_embedding_file_names = []
        self.embeddings_dir = os.path.join(self.root_dir, 'embeddings_6144')
        
        for root, dirs, files in os.walk(self.embeddings_dir):
            for file in files:
                if file.endswith(".npy"):
                     self.list_of_embedding_file_names.append(file)
        
        list_of_spectrogram_file_names = []
        self.spectrograms_dir = os.path.join(self.root_dir, 'spectrograms')
        
        for root, dirs, files in os.walk(self.spectrograms_dir):
            for file in files:
                if file.endswith(".npy"):
                     list_of_spectrogram_file_names.append(file)
                        
        assert set(list_of_spectrogram_file_names) == set(self.list_of_embedding_file_names)
        
        del list_of_spectrogram_file_names
            

    def __len__(self):
        return len(self.list_of_embedding_file_names)

    def __getitem__(self, idx):
        
        file_name = self.list_of_embedding_file_names[idx]
        
        emb_path = os.path.join(self.embeddings_dir, file_name)
        
        spec_path = os.path.join(self.spectrograms_dir, file_name)
        
        with open(emb_path, 'rb') as f:
            emb = np.load(f)
            
        with open(spec_path, 'rb') as f:
            spec = np.load(f)
            
        n = emb.shape[0]
        i = random.randrange(n)
        
        emb_tensor = torch.from_numpy(emb[i])
#         print(spec[i].shape)
        spec_tensor = torch.from_numpy(spec[i]).permute(2, 0, 1)
        
        return emb_tensor, spec_tensor, torch.tensor(i)
