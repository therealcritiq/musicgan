import tensorflow as tf
import math

sample_rate = 16000
hop_length_seconds=0.010

def load_real_mel_spec():
    for waveform in data:
        window_length_seconds = None # TODO: calculate this from waveform
        window_length_samples = int(round(window_length_seconds * sample_rate))
        hop_length_samples = int(round(hop_length_seconds * sample_rate))
        fft_length = 2**int(
            math.ceil(math.log(window_length_samples) / math.log(2.0)))
        tf.signal.stft(
            waveform,
            frame_length=1024,
            frame_step=256,
            # frame_step=hop_length_samples,
            fft_length=fft_length
        )
    pass