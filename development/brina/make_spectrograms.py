import os
import numpy as np
import math
import tensorflow as tf
import openl3
import soundfile as sf
from openl3.core import _preprocess_audio_batch
from kapre.time_frequency import Spectrogram, Melspectrogram
from keras.layers import Input
from keras.models import Model, Sequential
from tensorflow.python.keras.backend import set_session
import keras.backend as K

data_dir = "/scratch/prs392/incubator/data/LibriSpeech/test-clean/"
data_write_dir = "/scratch/prs392/incubator/data/LibriSpeech/test-clean/spectrograms"

# Get list of paths to audio files
list_of_file_paths = []
for root, dirs, files in os.walk(data_dir):
    path = root.split(os.sep)
    for file in files:
        if file.endswith(".flac"):
            list_of_file_paths.append(os.path.join(root, file))

# Get list of .npy (output) files that have already been created, so we can avoid doing it again 
list_of_npy_files = []
for root, dirs, files in os.walk(data_write_dir):
    path = root.split(os.sep)
    for file in files:
        if file.endswith(".npy"):
            list_of_npy_files.append(file.split('.')[0])


# Parameters for mel spectrogram (128 frequency bins)
# From _construct_mel128_audio_network at https://github.com/marl/openl3/blob/master/openl3/models.py
weight_decay = 1e-5
n_dft = 2048
n_mels = 128
n_hop = 242
asr = 48000
audio_window_dur = 1

# Start tf session
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
# Set up spectrogram model
x=Input(shape=(1, asr * audio_window_dur), dtype='float32')
mel = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
                              sr=asr, power_melgram=1.0, htk=True, # n_win=n_win,
                              return_decibel_melgram=True, padding='same')(x)
model = Model(inputs=x, outputs=mel)
tf.global_variables_initializer()
sess.run(tf.initialize_all_variables())

spectrograms_created = 0

for i, audio_file_path in enumerate(list_of_file_paths):

    file_name = os.path.basename(audio_file_path)

    file_id = file_name.split('.')[0]

    if file_id not in list_of_npy_files:

        audio, sr  = sf.read(audio_file_path)

        # Pre-process with padding, centering, hop size, etc. 
        audio_prep = _preprocess_audio_batch(audio, sr, center=False, hop_size=0.1)

        # Break up into batches of size 100 if needed
        batch_size=32
        #  Do the mel spectrogram transformation
        #audio_prep_tensor = tf.constant(audio_prep, dtype='float32')
        audio_mel = model.predict(audio_prep)

        # Save result
        new_spec_file_name = file_id + ".npy"
        new_spec_file_path = os.path.join(data_write_dir, new_spec_file_name)
        with open(new_spec_file_path, 'wb+') as f:
            np.save(f, audio_mel)
        spectrograms_created = spectrograms_created + 1

        # Every 1000 cases, reset tenserflow session (to avoid memory leak)
        if spectrograms_created == 1000:
            K.clear_session()
            sess = tf.Session()
            graph = tf.get_default_graph()
            set_session(sess)
            # Set up spectrogram model
            x=Input(shape=(1, asr * audio_window_dur), dtype='float32')
            mel = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
                                  sr=asr, power_melgram=1.0, htk=True, # n_win=n_win,
                                  return_decibel_melgram=True, padding='same')(x)
            model = Model(inputs=x, outputs=mel)
            tf.global_variables_initializer()
            spectrograms_created = 0

    else:
       #print(file_id + '.npy already exists') 
       pass

    # Log progress
    if i%1000 == 0:
        print("{} of {} spectrograms saved.".format(i+1, len(list_of_file_paths)+1))


