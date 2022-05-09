import tensorflow as tf
from tensorflow import keras
from keras import layers
import tf_slim
from base_model import BaseModel, CustomLambda
from utils import get_final_resolutions, get_kernel_scales, get_scale_factor, he_initializer_scale, get_num_filters, upscale_input
from activations import pixel_norm
import hparams
import utils


def tempfn(x):
    z = utils.minibatch_mean_stddev(x)
    y = utils.scalar_concat(x, z)
    # print('WTFFFF : -----------> ', x.shape, z, y.shape)
    return y

class Discriminator(BaseModel):
    def __init__(self, latent_vector_size=hparams.latent_vector_size, num_blocks=hparams.num_resolutions) -> None:
        super(Discriminator, self).__init__(save_input_to_lods=True)
        self.lods = []
        self.num_blocks = num_blocks
        # self.z = tf.concat(inputs, axis=1)
        # self.initial_x = tf_slim.flatten(self.z)
        # self.batch_size = int(self.z.shape[0])
        self.latent_vector_size = latent_vector_size
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.Input(shape=(128, 1024, 2)),
            self._conv2d(kernel_size=1, filters=utils.get_num_filters(0), name="conv_initial")[0],
            *self._compose_downsample_conv2d_blocks(num_blocks=self.num_blocks),
            CustomLambda(tempfn),
            self._conv2d(kernel_size=hparams.kernel_size,
                    filters=utils.get_num_filters(1),
                    name="conv_final-1",
                    activation=tf.nn.leaky_relu
            )[0],
            self._conv2d(kernel_size=hparams.kernel_size,
                filters=utils.get_num_filters(0),
                name="conv_final-2",
                # padding='VALID',
                activation=tf.nn.leaky_relu
            )[0],
            CustomLambda(self._get_dense)
        ])
        return model
    
    def _get_dense(self, x):
        # print('---x shape : -------------------- ', x.shape.as_list()[-1], ' and ---- ', x.shape[-1])
        flattened_x = layers.Flatten()(x)
        init_scale, post_scale = get_kernel_scales(
            kernel_shape=(x.shape.as_list()[-1], 1),
            he_initializer_slope=1.0,
            use_weight_scaling=True
        )
        bias = tf.zeros(shape=(3, ), dtype=tf.float32, name=f"discriminator-dense-bias")
        dense_output = layers.Dense(
            units=1,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=init_scale))(flattened_x)
        return post_scale * dense_output + bias