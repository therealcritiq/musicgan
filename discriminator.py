import tensorflow as tf
from tensorflow import keras
from keras import layers
import tf_slim
from base_model import BaseModel, CustomLambda
from utils import get_final_resolutions, get_kernel_scales, get_scale_factor, he_initializer_scale, get_num_filters, upscale_input
from activations import pixel_norm
import hparams


class Discriminator(BaseModel):
    def __init__(self, inputs, num_blocks=hparams.num_resolutions) -> None:
        self.lods = []
        self.num_blocks = num_blocks
        self.z = tf.concat(inputs, axis=1)
        self.initial_x = tf_slim.flatten(self.z)
        self.batch_size = int(self.z.shape[0])
        self.model = self._build_model(self.initial_x)

    def _build_model(self, init_x):
        model = keras.Sequential([
            keras.Input(shape=init_x.shape[1]),
            self._conv2d(kernel_size=(1, 1), filters=32, name="conv_final")[0],
            *self._compose_downsample_conv2d_blocks(num_blocks=self.num_blocks),
            CustomLambda(self._get_dense)
        ])
    
    def _get_dense(self, x):
        # print('---x shape : -------------------- ', x.shape.as_list()[-1], ' and ---- ', x.shape[-1])
        flattened_x = layers.Flatten()(x)
        init_scale, post_scale = get_kernel_scales(
            kernel_shape=(x.shape.as_list()[-1], 1),
            he_initializer_slope=1.0,
            use_weight_scaling=True
        )
        bias = tf.zeros(shape=(3, ), dtype=tf.float32, name=f"{self.name}-bias")
        dense_output = layers.Dense(
            units=1,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=init_scale))(flattened_x)
        return post_scale * dense_output + bias
    
    