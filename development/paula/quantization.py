import os
import numpy as np
import audio_dataset as data_utils

AudioDataset = data_utils.AudioDataset
audio_dataset = AudioDataset(root_dir='/scratch/kk4158/incubator/data/LibriSpeech/train-clean-360')

data_write_dir = "/scratch/kk4158/incubator/data/LibriSpeech/train-clean-360/quantized_8bit"

num_file = 0     # for debugging


# pca param
pca_matrix = np.load('/scratch/kk4158/incubator/code/pca_matrix.npy')
pca_means = np.load('/scratch/kk4158/incubator/code/pca_means.npy')
pca_means = pca_means.reshape(-1, 1)

# quantize param
QUANTIZE_MIN_VAL = -2
QUANTIZE_MAX_VAL = 2

curr_file = None
list_of_embs = []



for i in range(0, len(audio_dataset)):
    emb, file_name = audio_dataset[i]

    if not curr_file:
    	curr_file = file_name

    if curr_file == file_name:      # same file
    	list_of_embs.append(emb)

    else:							# end of file
    	emb_batch = np.array(list_of_embs)
    	# apply pca
    	pca_applied = np.dot(pca_matrix, (emb_batch.T - pca_means)).T
		# clipping to [min, max] range
    	clipped_emb = np.clip(pca_applied, QUANTIZE_MIN_VAL, QUANTIZE_MAX_VAL)
    	# convert to 8--bit in range [0.0, 255.0]
    	quantized_emb = ((clipped_emb - QUANTIZE_MIN_VAL) * (255.0 / (QUANTIZE_MAX_VAL - QUANTIZE_MIN_VAL)))
    	# cast 8-bit float to uint8
    	quantized_emb = quantized_emb.astype(np.uint8)
	    # save the quantized embeddings to directory
    	quan_emb_file_path = os.path.join(data_write_dir, curr_file)
    	with open(quan_emb_file_path, 'wb+') as f:
    		np.save(f, quantized_emb)
    	### debugging ###
    	num_file += 1
    	# print('-' * 20)
    	# print(i, num_file, curr_file, quantized_emb.shape)
    	# print('-' * 20)
    	n, d = emb_batch.shape
    	file = open('quan8.txt', 'a')
    	file.write("%i %i %s (%i, %i)\n" % (i, num_file, curr_file, n, d))
    	file.close
    	#################

    	# start of new file
    	curr_file = file_name
    	list_of_embs = []
    	list_of_embs.append(emb)

    # last embedding
    if i == len(audio_dataset)-1:
    	emb_batch = np.array(list_of_embs)
    	# apply pca
    	pca_applied = np.dot(pca_matrix, (emb_batch.T - pca_means)).T
		# clipping to [min, max] range
    	clipped_emb = np.clip(pca_applied, QUANTIZE_MIN_VAL, QUANTIZE_MAX_VAL)
    	# convert to 8--bit in range [0.0, 255.0]
    	quantized_emb = ((clipped_emb - QUANTIZE_MIN_VAL) * (255.0 / (QUANTIZE_MAX_VAL - QUANTIZE_MIN_VAL)))
    	# cast 8-bit float to uint8
    	quantized_emb = quantized_emb.astype(np.uint8)
	    # save the quantized embeddings to directory
    	quan_emb_file_path = os.path.join(data_write_dir, curr_file)
    	with open(quan_emb_file_path, 'wb+') as f:
    		np.save(f, quantized_emb)
    	### debugging ###
    	num_file += 1
    	# print('-' * 20)
    	# print(i, num_file, curr_file, quantized_emb.shape)
    	# print('-' * 20)
    	n, d = emb_batch.shape
    	file = open('quan8.txt', 'a')
    	file.write("%i %i %s (%i, %i)\n" % (i, num_file, curr_file, n, d))
    	file.close
    	#################


