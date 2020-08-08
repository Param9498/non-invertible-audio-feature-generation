# downstream task pipline

## Classification Problem: 
Speaker Identification on 30 speakers


## Goal:
Maintain  high accuracy score  even with heavy counter measures

## Data:
30 speakers’ audio book recordings from LibriSpeech dataset

Train:  ~80 audio clips  for each speaker, we select  ~140  1s audio frame from each audio clip

Test:   ~16 audio clips  for each speaker, we select  ~140  1s audio frame from each audio clip

access the data on NYU HPC (please request access first) /scratch/prs392/incubator/data/speaker_identification/

## Model:
RandomForest, Multi-layered Perceptron
