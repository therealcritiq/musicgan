from os import listdir
from os.path import isfile, join
import librosa
import librosa.display
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import numpy as np

sample_rate = 16000
hop_length_seconds=0.010
window_length_seconds = 1 # default length of each audio one-shot in seconds

def display_mel_spec(mel_spec):
    print('displaying mel spec', mel_spec)
    plt.figure()
    librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max))
    plt.colorbar()
    plt.show()

def get_dir_files(dirpath='./data'):
    return [f for f in listdir(dirpath) if isfile(join(dirpath, f))]


def load_audio_to_mel_spec(dirpath='./data'):
    mel_specs = []
    filepaths = get_dir_files(dirpath)
    print('filepaths found : ', filepaths)
    for audiofile in filepaths:
        # Note: signal is converted to mono by default
        x, sr = librosa.load(f'{dirpath}/{audiofile}', sr=None)
        print('got file at sr: ', sr)
        window_length_samples = int(round(window_length_seconds * sample_rate))
        fft_length = 2**int(
            math.ceil(math.log(window_length_samples) / math.log(2.0)))
        mel_specs.append(
            librosa.feature.melspectrogram(
                x,
                sr=sr,
                n_fft=fft_length,
                hop_length=256,
                win_length=1024
            )
        )
    return mel_specs