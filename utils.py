import numpy as np
import tensorflow as tf
import pitch_counts
import hparams

def he_initializer_scale(shape, slope=1.0):
    """The scale of He neural network initializer.
    Args:
        shape: A list of ints representing the dimensions of a tensor.
        slope: A float representing the slope of the ReLu following the layer.
    Returns:
        A float of he initializer scale.
    """
    fan_in = np.prod(shape[:-1])
    return np.sqrt(2. / ((1. + slope**2) * fan_in))

# https://github.com/magenta/magenta/blob/f73ff0c91f0159a925fb6547612199bb7c915248/magenta/models/gansynth/lib/train_util.py#L144
def generate_latent_vector_z(batch_size, latent_vector_size=hparams.latent_vector_size):
    """Returns a batch of `batch_size` random latent vectors."""
    return tf.random_normal([batch_size, latent_vector_size], dtype=tf.float32)

# https://github.com/magenta/magenta/blob/f73ff0c91f0159a925fb6547612199bb7c915248/magenta/models/gansynth/lib/datasets.py#L83
def get_pitch_one_hot_labels(batch_size):
    """Provides one hot labels."""
    pitches = sorted(pitch_counts.keys())
    counts = [pitch_counts[p] for p in pitches]
    indices = tf.reshape(
        tf.multinomial(tf.log([tf.to_float(counts)]), batch_size), [batch_size])
    one_hot_labels = tf.one_hot(indices, depth=len(pitches))
    return one_hot_labels
