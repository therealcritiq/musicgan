import tensorflow as tf
from tensorflow import keras
from keras import layers
from utils import he_initializer_scale, get_num_filters
from activations import pixel_norm
import hparams

class Generator():
    def __init__(self, inputs, num_blocks=hparams.num_resolutions) -> None:
        self.num_blocks = num_blocks
        self.z = tf.concat(inputs, axis=1)
        self.model = self._build_model(self.z)
    
    def _build_model(self, x):
        """
        TODO: NEED TO FIX THE PARAMETERS FOR EACH OF THE CONV2D BLOCKS
        TODO: update arch to mimic: https://github.com/magenta/magenta/blob/f73ff0c91f0159a925fb6547612199bb7c915248/magenta/models/gansynth/lib/networks.py#L362
        """
        model = keras.Sequential([
            keras.Input(shape=x.shape), # TODO: need to fix this input shape to make sure it includes the num of filters/channels
            *self._conv2d_doubled(
                kernel_size=hparams.kernel_size,
                filters=get_num_filters(1),
                padding="SAME"
            ),
            *self._compose_upsample_conv2d_blocks(num_blocks=self.num_blocks),
            # TODO: OUTPUT DENSE 
        ])
        self.model = model
    
    def _compose_upsample_conv2d_blocks(self, num_blocks=1):
        blocks = []
        for i in range(1, num_blocks):
            blocks.extend(
                self._upsample_conv2d_block(
                    conv2d_params={
                        "kernel_size":self.kernel_size,
                        "filters": get_num_filters(i + 1),
                        "padding": "SAME"
                    },
                    upsample_params={
                        "block_id": i + 1
                    }
                )
            )
        return blocks

    def _conv2d(self,
                filters,
                kernel_size,
                strides=(1, 1),
                padding='SAME',
                he_initializer_slope=1.0,
                use_weight_scaling=True):
        def build_conv2d_and_scale(input):
            kernel_shape = kernel_size + [input.shape.as_list()[3], filters],
            kernel_scale = he_initializer_scale(kernel_shape, he_initializer_slope)
            init_scale, post_scale = kernel_scale, 1.0
            if use_weight_scaling:
                init_scale, post_scale = post_scale, init_scale
            
            kernel_initializer = tf.random_normal_initializer(stddev=init_scale)

            return [
                layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    kernel_initializer=kernel_initializer
                ),
                layers.Lambda(lambda kernel_output: post_scale * kernel_output)
            ]

        return [
            layers.Lambda(build_conv2d_and_scale),
            layers.Lambda(lambda x: x[1]),
            layers.Lambda(lambda x: pixel_norm(tf.nn.leaky_relu(x)))
        ]

    def _conv2d_doubled(self, **kwargs):
        return [
            *self._conv2d(**kwargs),
            *self._conv2d(**kwargs),
        ]
    
    def _upsample(self, **kwargs):
        scale = hparams.scale_base**(self.num_blocks - kwargs.get('block_id', 1))
        return layers.Lambda(
            lambda images: tf.batch_to_space(
                    tf.tile(images, [scale**2, 1, 1, 1]),
                    crops=[[0, 0], [0, 0]],
                    block_size=scale
                )
            )
    
    def _upsample_conv2d_block(self, conv2d_params, upsample_params):
        return [
            self._upsample(**upsample_params),
            *self._conv2d_doubled(**conv2d_params)
        ]