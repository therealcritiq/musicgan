from turtle import shape
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tf_slim
from utils import generator_alpha, get_final_resolutions, get_scale_factor, he_initializer_scale, get_num_filters, upscale_input
from activations import pixel_norm
import hparams

class Generator():
    def __init__(self, inputs, num_blocks=hparams.num_resolutions) -> None:
        self.lods = []
        self.num_blocks = num_blocks
        self.z = tf.concat(inputs, axis=1)
        self.initial_x = tf_slim.flatten(self.z)
        self.batch_size = int(self.z.shape[0])
        self.model = self._build_model(self.initial_x)
    
    def _build_model(self, init_x):
        """
        TODO: NEED TO FIX THE PARAMETERS FOR EACH OF THE CONV2D BLOCKS
        TODO: update arch to mimic: https://github.com/magenta/magenta/blob/f73ff0c91f0159a925fb6547612199bb7c915248/magenta/models/gansynth/lib/networks.py#L362
        """
        final_h, final_w = get_final_resolutions()
        model = keras.Sequential([
            keras.Input(shape=init_x.shape[1]), # TODO: need to fix this input shape to make sure it includes the num of filters/channels
            CustomLambda(lambda x: tf.expand_dims(tf.expand_dims(x, 1), 1), name="expand_dims"),
            CustomLambda(pixel_norm, name="pixel_norm_1"),
            CustomLambda(
                lambda x: tf.pad(
                    x, 
                    # The padding that's defined in the GANSynth Github code 
                    # doesn't let us use a kernel_size of (6, 2) in order to match the output architecture
                    # as defined in the GANSynth paper here: https://arxiv.org/pdf/1902.08710.pdf
                    # [[0] * 2, [hparams.start_resolutions[0] - 1] * 2,
                    # [hparams.start_resolutions[1] - 1] * 2, [0] * 2]
                    [[0,0], [3, 3], [8,8], [0,0]]
                ),
                name="pad_1"
            ),
            # kernel_size was previously just hparams.start_resolutions based off of what the GANSynth Github code seemed to show
            # but the output of each layer using (4, 4) doesn't give us the same output architecture as defined in the
            # GANSynth paper
            # The following kernel_size of (6, 2) does
            *self._conv2d(kernel_size=(6, 2), filters=get_num_filters(1), padding="VALID", name="conv2d_1"),
            *self._conv2d(filters=get_num_filters(1)),
            *self._compose_upsample_conv2d_blocks(num_blocks=self.num_blocks),
            # TODO: OUTPUT FINAL TANH LAYER
        ])
        self.model = model
        return model
    
    def _compose_upsample_conv2d_blocks(self, num_blocks=1):
        blocks = []
        for i in range(1, num_blocks):
            blocks.extend(
                self._upsample_conv2d_block(
                    conv2d_params={
                        "filters": get_num_filters(i + 1),
                        "padding": "SAME",
                        "name": f"UPSAMPLE_CONV_BLOCK_{i}-conv"
                    },
                    upsample_params={
                        "block_id": i + 1,
                        "name": f"UPSAMPLE_CONV_BLOCK_{i}-upsample"
                    }
                )
            )
            blocks.append(CustomLambda(self._save_input_to_lods, name=f"UPSAMPLE_CONV_BLOCK_{i}-lods"))
        return blocks

    def _conv2d(self,
                filters=3,
                kernel_size=hparams.kernel_size,
                strides=(1, 1),
                padding='SAME',
                he_initializer_slope=1.0,
                use_weight_scaling=True,
                name=None):
        return [
            CustomConv2D(
                filters,
                kernel_size,
                strides,
                padding,
                he_initializer_slope,
                use_weight_scaling,
                name=name
            ),
            CustomLambda(lambda x: pixel_norm(tf.nn.leaky_relu(x)), name=f"{name}-pixel_norm")
        ]

    def _conv2d_doubled(self, name, **kwargs):
        return [
            *self._conv2d(**kwargs, name=f"{name}-1"),
            *self._conv2d(**kwargs, name=f"{name}-2"),
        ]
    
    def _compose_rgb_blocks(self, num_blocks=1):
        # https://github.com/magenta/magenta/blob/f73ff0c91f0159a925fb6547612199bb7c915248/magenta/models/gansynth/lib/networks.py#L397
        # outputs = []
        # for block_id in range(num_blocks):
        #     lod = layers.custom_conv2d(
        #             x=x,
        #             filters=colors,
        #             kernel_size=1,
        #             padding='SAME',
        #             activation=to_rgb_activation,
        #             scope='to_rgb')
        #     scale = get_scale_factor(block_id)
        #     lod = upscale_input(lod, scale)
        #     alpha = generator_alpha(block_id, progress)
        #     outputs.append(lod * alpha)
        pass

    def _save_input_to_lods(self, x):
        self.lods.append(x)
        return x

    def _upsample(self, **kwargs):
        name = kwargs.get("name") or None
        return CustomLambda(upscale_input, name=name)
    
    def _upsample_conv2d_block(self, conv2d_params, upsample_params):
        return [
            self._upsample(**upsample_params),
            *self._conv2d_doubled(**conv2d_params)
        ]


class CustomConv2D(layers.Layer):
    def __init__(self,
                filters=3,
                kernel_size=3,
                strides=(1, 1),
                padding="SAME",
                he_initializer_slope=1.0,
                use_weight_scaling=True,
                **kwargs):
        super(CustomConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.he_initializer_slope = he_initializer_slope
        self.use_weight_scaling = use_weight_scaling

    def call(self, inputs):
        # print('custom conv2d input shape :', inputs.shape)
        kernel_shape = [self.kernel_size] + [inputs.shape[3], self.filters],
        kernel_scale = he_initializer_scale(kernel_shape, self.he_initializer_slope)
        init_scale, post_scale = kernel_scale, 1.0
        if self.use_weight_scaling:
            init_scale, post_scale = post_scale, init_scale
        
        kernel_initializer = tf.random_normal_initializer(stddev=init_scale)
        x = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=kernel_initializer,
            name=f"{self.name}-conv"
        )(inputs)

        bias = tf.zeros(shape=(self.filters, ), dtype=tf.float32, name=f"{self.name}-bias")
        x = CustomLambda(lambda kernel_output: post_scale * kernel_output + bias)(x)
        self.inputs = x
        return self.inputs

    def get_config(self):
        config = super(CustomConv2D, self).get_config()
        return config

class CustomLambda(layers.Lambda):
    def __init__(self, function, output_shape=None, mask=None, arguments=None, **kwargs):
        super(CustomLambda, self).__init__(function, output_shape, mask, arguments, **kwargs)
        self.lambda_fn = function
    
    def call(self, inputs):
        # print(f"Lambda [{self.name}] input shape : ", inputs.shape)
        return self.lambda_fn(inputs)