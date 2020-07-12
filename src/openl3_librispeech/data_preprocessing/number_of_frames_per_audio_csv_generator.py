import os
import sys
import numpy as np

import pandas as pd

list_of_embedding_file_names = {}

folder_names = ['train-clean-360', 'dev-clean', 'test-clean']

for folder in folder_names:

    root_dir = os.path.join('/scratch/prs392/incubator/data/LibriSpeech/', folder)

    for root, dirs, files in os.walk(os.path.join(root_dir, 'embeddings_6144')):

        for file in files:

            if file.endswith(".npy"):

                if file in list_of_embedding_file_names:
                    print("Repeating files. Shutting down.")
                    list_of_embedding_file_names = {}
                    sys.exit(0)

                emb_path = os.path.join(root, file)

                with open(emb_path, 'rb') as f:
                    emb = np.load(f)
                list_of_embedding_file_names[file] = emb.shape[0]
                
    df = pd.DataFrame({'file_name' : list(list_of_embedding_file_names.keys()), 'number_of_frames' : list(list_of_embedding_file_names.values())})
    df.to_csv(os.path.join(root_dir, 'number_of_frames_per_audio.csv'), index = False)
    
    with open(os.path.join(root_dir, 'number_of_frames_per_audio.npy'), 'wb') as f:
        np.save(f, df.values)