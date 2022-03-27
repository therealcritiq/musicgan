import tensorflow as tf

# https://github.com/magenta/magenta/blob/f73ff0c91f0159a925fb6547612199bb7c915248/magenta/models/gansynth/lib/layers.py#L30
def pixel_norm(images, epsilon=1.0e-8):
    """Pixel normalization.
    For each pixel a[i,j,k] of image in HWC format, normalize its value to
    b[i,j,k] = a[i,j,k] / SQRT(SUM_k(a[i,j,k]^2) / C + eps).
    Args:
    images: A 4D `Tensor` of NHWC format.
    epsilon: A small positive number to avoid division by zero.
    Returns:
    A 4D `Tensor` with pixel-wise normalized channels.
    """
    return images * tf.rsqrt(
        tf.reduce_mean(tf.square(images), axis=3, keepdims=True) + epsilon
    )