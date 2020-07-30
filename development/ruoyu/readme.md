### downstream task pipline

dataset: OpenMIC-2018 

https://zenodo.org/record/1432913#.XyMaBpNKhhA

the dataset 
10 second snippets of audio















dataset: GTZAN Genre Collection

http://opihi.cs.uvic.ca/sound/genres.tar.gz

The dataset consists of 1000 = 10 genres x 100 audio track, each 30 seconds long. (30s * 1000 * 1 = 30,000s)

genre: blues, classical,country, disco, etc

The tracks are all 22050 Hz monophonic 16-bit audio files in .au format.

Feature extractor: Openl3

Classification model: linearSVM (a multi-class one-vs-all linear SVM is trained) as in the openl3 paper

#### feature extractor: model_type == 'openl3',  hop_size = 1, embedding_size = 6144, content_type = 'music'

model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music", embedding_size=6144) /512

wave, sr = wavefile_to_waveform(config['audio_folder'] + p, 'openl3')
            
emb, _ = openl3.get_audio_embedding(wave, sr, hop_size=1, model=model, verbose=False)
            
            
            
            
#### benchmark: testing result 

Accuracy: 0.7310344827586207


Confusion matrix:

[[19  0  0  0  0  5  0  0  1  6] blues    
 [ 0 31  0  0  0  0  0  0  0  0] classical  
 [ 1  0 22  5  0  0  0  2  0  0] country   
 [ 0  0  0 25  3  0  0  1  0  0] discov    
 [ 0  0  0  5 19  0  0  3  0  0] hipop     
 [ 0  0  0  0  0 25  0  1  0  1] jazz      
 [ 0  0  0  0  0  0 27  0  0  0] metal     
 [ 0  0  0  5  0  0  0 24  1  0] pop       
 [ 0  0  0  3  4  0  0  2 16  1] reggae    
 [ 5  2  2 10  5  0  1  2  1  4]] rock      





