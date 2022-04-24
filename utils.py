import math
import numpy as np
import tensorflow as tf
from pitch_counts import pitch_counts
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

# https://github.com/magenta/magenta/blob/f73ff0c91f0159a925fb6547612199bb7c915248/magenta/models/gansynth/lib/networks.py#L206
def generator_alpha(block_id, progress):
    return tf.maximum(0.0,
        tf.minimum(progress - (block_id - 2), block_id - progress))

def upscale_input(input, scale=hparams.scale_base):
    return tf.batch_to_space(
        tf.tile(input, [scale**2, 1, 1, 1]),
        # tf.tile(images, [scale**2, 1, 1, 1]),
        crops=[[0, 0], [0, 0]],
        block_shape=scale
    )

def downscale_input(input, scale=hparams.scale_base):
    return tf.nn.avg_pool(
      input,
      ksize=[1, scale, scale, 1],
      strides=[1, scale, scale, 1],
      padding='VALID')

def get_scale_factor(block_id):
    return hparams.scale_base**(hparams.num_resolutions - block_id)

def get_final_resolutions():
    return tuple(r * get_scale_factor(1) for r in hparams.start_resolutions)

# https://github.com/magenta/magenta/blob/f73ff0c91f0159a925fb6547612199bb7c915248/magenta/models/gansynth/lib/networks.py#L285
def get_num_filters(block_id, fmap_base=4096, fmap_decay=1.0, fmap_max=256):
    """Computes number of filters of block `block_id`."""
    num_filters = int(min(fmap_base / math.pow(2.0, block_id * fmap_decay), fmap_max))
    print(f"Num filters calculated  for block id {block_id}: {num_filters}")
    return num_filters

# https://github.com/magenta/magenta/blob/f73ff0c91f0159a925fb6547612199bb7c915248/magenta/models/gansynth/lib/train_util.py#L144
def sample_random_noise_vector(batch_size, latent_vector_size=hparams.latent_vector_size):
    """Returns a batch of `batch_size` random latent vectors."""
    return tf.random.normal([batch_size, latent_vector_size], dtype=tf.float32)

# https://github.com/magenta/magenta/blob/f73ff0c91f0159a925fb6547612199bb7c915248/magenta/models/gansynth/lib/datasets.py#L83
def get_pitch_one_hot_labels(batch_size):
    """Provides one hot labels."""
    pitches = sorted(pitch_counts.keys())
    counts = [pitch_counts[p] for p in pitches]
    indices = tf.reshape(
        tf.random.categorical(tf.math.log([tf.cast(counts, tf.float32)]), batch_size), [batch_size])
    one_hot_labels = tf.one_hot(indices, depth=len(pitches), dtype=tf.float32)
    return one_hot_labels

def get_kernel_scales(
    kernel_shape=None,
    he_initializer_slope=None,
    use_weight_scaling=False
):
    kernel_scale = he_initializer_scale(kernel_shape, he_initializer_slope)
    init_scale, post_scale = kernel_scale, 1.0
    if use_weight_scaling:
        init_scale, post_scale = post_scale, init_scale
    return init_scale, post_scale

# https://github.com/magenta/magenta/blob/f73ff0c91f0159a925fb6547612199bb7c915248/magenta/models/gansynth/lib/networks.py#L256
def blend_images(x, progress, num_blocks):
  """Blends images of different resolutions according to `progress`.
  When training `progress` is at a stable stage for resolution r, returns
  image `x` downscaled to resolution r and then upscaled to `final_resolutions`,
  call it x'(r).
  Otherwise when training `progress` is at a transition stage from resolution
  r to 2r, returns a linear combination of x'(r) and x'(2r).
  Args:
    x: An image `Tensor` of NHWC format with resolution `final_resolutions`.
    progress: A scalar float `Tensor` of training progress.
    resolution_schedule: An object of `ResolutionSchedule`.
    num_blocks: An integer of number of blocks.
  Returns:
    An image `Tensor` which is a blend of images of different resolutions.
  """
  x_blend = []
  for block_id in range(1, num_blocks + 1):
    alpha = generator_alpha(block_id, progress)
    scale = get_scale_factor(block_id)
    rescaled_x = upscale_input(
        downscale_input(x, scale), scale)
    x_blend.append(alpha * rescaled_x)
  return tf.add_n(x_blend)