from matplotlib import pyplot as plt
import IPython.display as ipd
import random

def print_plot_play(x, Fs, text=''):
    """1. Prints information about an audio singal, 2. plots the waveform, and 3. Creates player
    
    Notebook: C1/B_PythonAudio.ipynb
    
    Args: 
        x: Input signal
        Fs: Sampling rate of x    
        text: Text to print
    """
    print('%s Fs = %d, x.shape = %s, x.dtype = %s' % (text, Fs, x.shape, x.dtype))
    plt.figure(figsize=(10, 2))
    plt.plot(x, color='gray')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    ipd.display(ipd.Audio(data=x, rate=Fs))

def sample_one_second(audio_data, sampling_frequency, augment=False):
    """Return one second audio samples randomly
    Args:
        audio_data: audio data to sample from
        sampling_frequency: audio sample rate
        augment: if True, perturb the data in some fashion
    Returns:
        One second samples, start time, and augmentation parameters
    """
    sampling_frequency = int(sampling_frequency)
    if len(audio_data) > sampling_frequency:
        start = random.randrange(len(audio_data) - sampling_frequency)
    else:
        start = 0

    audio_data = audio_data[start:start+sampling_frequency]

    if audio_data.shape[0] != sampling_frequency:
        # Pad audio that isn't one second
        warnings.warn('Got audio that is less than one second', UserWarning)
        
        audio_data = np.pad(audio_data,
                            ((0, sampling_frequency - audio_data.shape[0]),),
                            mode='constant')

    if augment:
        orig_dtype = audio_data.dtype
        audio_data = audio_data.astype(float)
        # Make sure we don't clip
        if np.abs(audio_data).max():
            max_gain = min(0.1, get_max_abs_sample_value(orig_dtype)/np.abs(audio_data).max() - 1)
        else:
            # TODO: Handle audio with all zeros
            warnings.warn('Got audio sample with all zeros', UserWarning)
            max_gain = 0.1
        gain = 1 + random.uniform(-0.1, max_gain)
        assert 0.9 <= gain <= 1.1
        audio_data *= gain

        audio_data = audio_data.astype(orig_dtype)
        audio_aug_params = {'gain': gain}
    else:
        audio_aug_params = {}

    return audio_data, start, audio_aug_params