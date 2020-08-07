import os
import random

import torch
from torch.utils.data import Dataset
import torchaudio.functional as F

import numpy as np


class AudioDataset(Dataset):

    def __init__(self, root_dir, transform=None, num_audios = -1):
        
        self.root_dir = root_dir
        self.transform = transform
        
        self.num_audios = num_audios
        
        self.list_of_embedding_file_names = []
        self.embeddings_dir = os.path.join(self.root_dir, 'embeddings_6144')
        
#         print(num_audios)
        
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
        
#         print(len(set(list_of_spectrogram_file_names)), len(set(self.list_of_embedding_file_names)))
#         print(np.setdiff1d(self.list_of_embedding_file_names,list_of_spectrogram_file_names))
        assert set(list_of_spectrogram_file_names) == set(self.list_of_embedding_file_names)
        
        del list_of_spectrogram_file_names
        
        self.list_of_embedding_frames = []
        
        for i, file_name in enumerate(self.list_of_embedding_file_names):
            emb_path = os.path.join(self.embeddings_dir, file_name)
            temp = np.load(emb_path, mmap_mode='r')
            self.list_of_embedding_frames.append(temp.shape[0])
            if i == int(self.num_audios) - 1:
                break
                
        self.list_of_embedding_files_frames = [(self.list_of_embedding_file_names[i], j) 
                                               for i in range(len(self.list_of_embedding_frames))
                                               for j in range(self.list_of_embedding_frames[i]) ]
        
        
#         print(self.list_of_embedding_frames, len(self.list_of_embedding_files_frames))
                
    def __len__(self):
        return len(self.list_of_embedding_files_frames)

    def __getitem__(self, idx):
        
        file_name = self.list_of_embedding_files_frames[idx][0]
        
        emb_path = os.path.join(self.embeddings_dir, file_name)
        
        spec_path = os.path.join(self.spectrograms_dir, file_name)
        
        frame_idx = self.list_of_embedding_files_frames[idx][1]
        
        
        with open(emb_path, 'rb') as f:
            emb = np.load(f)
            
        with open(spec_path, 'rb') as f:
            spec = np.load(f)
            
#         print(self.list_of_embedding_files_frames[idx],frame_idx,len(emb),len(spec))
        
        emb_tensor = torch.from_numpy(emb[frame_idx])
        spec_tensor = torch.from_numpy(spec[frame_idx]).permute(2, 0, 1)
        
        spec_tensor_amp = F.DB_to_amplitude(x = spec_tensor, ref = 1, power = 0.5)
        
        return emb_tensor, spec_tensor_amp, torch.tensor(frame_idx)

