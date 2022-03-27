import numpy as np

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