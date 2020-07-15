### downsteam task pipline

dataset: GTZAN Genre Collection
The dataset consists of 1000 audio tracks each 30 seconds long. 
It contains 10 genres, each represented by 100 tracks. 
The tracks are all 22050 Hz monophonic 16-bit audio files in .au format.

Feature extractor: Openl3

Classification model: linearSVM (a multi-class one-vs-all linear SVM is trained) as in the openl3 paper

#### feature extractor 
model_type == 'openl3':
            wave, sr = wavefile_to_waveform(config['audio_folder'] + p, 'openl3')
            emb, _ = openl3.get_embedding(wave, sr, hop_size=1, model=model, verbose=False)
