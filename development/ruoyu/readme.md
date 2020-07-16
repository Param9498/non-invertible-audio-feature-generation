### downstream task pipline

dataset: GTZAN Genre Collection

http://opihi.cs.uvic.ca/sound/genres.tar.gz

The dataset consists of 1000 = 10 genres x 100 audio track, each 30 seconds long. (30s * 1000 * 1 = 30,000s)

genre: blues, classical,country, disco, etc

The tracks are all 22050 Hz monophonic 16-bit audio files in .au format.

Feature extractor: Openl3

Classification model: linearSVM (a multi-class one-vs-all linear SVM is trained) as in the openl3 paper

#### feature extractor: hop_size = 1, embedding_size = 6144, content_type = 'music'
model_type == 'openl3':
            wave, sr = wavefile_to_waveform(config['audio_folder'] + p, 'openl3')
            emb, _ = openl3.get_audio_embedding(wave, sr, hop_size=1, model=model, verbose=False)
            
            
            
            
#### benchmark: testing result (TBA)



