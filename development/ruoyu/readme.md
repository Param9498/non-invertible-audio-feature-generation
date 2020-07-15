### downsteam task pipline

dataset: GTZAN Genre Collection
The dataset consists of 1000 audio tracks each 30 seconds long. 
It contains 10 genres, each represented by 100 tracks. 
The tracks are all 22050 Hz monophonic 16-bit audio files in .au format.

Feature extractor: Openl3

Classification model: linearSVM (a multi-class one-vs-all linear SVM is trained) as in the openl3 paper
