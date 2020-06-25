import os
​
import openl3
import soundfile as sf
​
import numpy as np
​
data_dir = "/scratch/prs392/incubator/data/LibriSpeech/train-clean-360/"
​
list_of_file_paths = []
for root, dirs, files in os.walk(data_dir):
    path = root.split(os.sep)
    for file in files:
        if file.endswith(".flac"):
#             print(os.path.basename(os.path.join(root, file)))
            list_of_file_paths.append(os.path.join(root, file))
    
data_write_dir = "/scratch/prs392/incubator/data/LibriSpeech/train-clean-360/embeddings_6144"
​
list_of_npy_files = []
for root, dirs, files in os.walk(data_write_dir):
    path = root.split(os.sep)
    for file in files:
        if file.endswith(".npy"):
#             print(os.path.basename(os.path.join(root, file)))
            list_of_npy_files.append(file.split('.')[0])
    
# print(list_of_npy_files)
model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music",
                                                 embedding_size=6144)
​
for audio_file_path in list_of_file_paths:
    
    file_name = os.path.basename(audio_file_path)
    
    file_id = file_name.split('.')[0]
    
    if file_id not in list_of_npy_files:
        
        audio, sr  = sf.read(audio_file_path)
        
#         emb, ts = openl3.get_audio_embedding(audio, sr, center = False)
        emb, ts = openl3.get_audio_embedding(audio, sr, center = False, model=model)
        new_emb_file_name = file_id + ".npy"
        new_emb_file_path = os.path.join(data_write_dir, new_emb_file_name)
​
        with open(new_emb_file_path, 'wb+') as f:
            np.save(f, emb)
    else:
        print(file_id + '.npy already exists') 