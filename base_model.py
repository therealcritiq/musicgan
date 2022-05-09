import tensorflow as tf
from keras import layers
from utils import downscale_input, get_kernel_scales, get_num_filters, upscale_input
from activations import pixel_norm
import hparams

class BaseModel():
    def __init__(self, save_input_to_lods=False) -> None:
        self.save_input_to_lods = save_input_to_lods

    def _compose_downsample_conv2d_blocks(self, num_blocks=1):
        blocks = []
        for i in range(num_blocks, 1, -1):
            blocks.extend(
                self._downsample_conv2d_block(
                    num_filters_a=get_num_filters(i),
                    num_filters_b=get_num_filters(i - 1),
                    conv2d_params={
                        "padding": "SAME",
                        "name": f"DOWNSAMPLE_CONV_BLOCK_{i}-conv"
                    },
                    downsample_params={
                        "block_id": i,
                        "name": f"DOWNSAMPLE_CONV_BLOCK_{i}-upsample"
                    }
                )
            )
            if self.save_input_to_lods is not None:
                blocks.append(CustomLambda(self._save_input_to_lods, name=f"DOWNSAMPLE_CONV_BLOCK_{i}-lods"))
        return blocks

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
            if self.save_input_to_lods is not None:
                blocks.append(CustomLambda(self._save_input_to_lods, name=f"UPSAMPLE_CONV_BLOCK_{i}-lods"))
        return blocks

    def _conv2d(self,
                filters=3,
                kernel_size=hparams.kernel_size,
                strides=(1, 1),
                padding='SAME',
                he_initializer_slope=1.0,
                use_weight_scaling=True,
                activation=None,
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
            CustomLambda(lambda x: activation(x), name=f"{name}-tanh") if activation is not None 
            else CustomLambda(lambda x: pixel_norm(tf.nn.leaky_relu(x)), name=f"{name}-pixel_norm"),
        ]

    def _conv2d_doubled(self, name, num_filters_a, num_filters_b, **kwargs):
        ret = []
        kwargs['filters'] = num_filters_a
        ret.extend(self._conv2d(**kwargs, name=f"{name}-1"))
        kwargs['filters'] = num_filters_b
        ret.extend(self._conv2d(**kwargs, name=f"{name}-2"))
        return ret
    
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
    
    def _downsample(self, **kwargs):
        name = kwargs.get("name") or None
        return CustomLambda(downscale_input, name=name)

    def _upsample(self, **kwargs):
        name = kwargs.get("name") or None
        return CustomLambda(upscale_input, name=name)

    def _downsample_conv2d_block(self, num_filters_a, num_filters_b, conv2d_params, downsample_params):
        return [
            *self._conv2d_doubled(
                **conv2d_params,
                num_filters_a=num_filters_a,
                num_filters_b=num_filters_b
            ),
            self._downsample(**downsample_params)
        ]
    def _upsample_conv2d_block(self, conv2d_params, upsample_params):
        return [
            self._upsample(**upsample_params),
            *self._conv2d_doubled(
                **conv2d_params,
                num_filters_a=conv2d_params['filters'],
                num_filters_b=conv2d_params['filters']
            )
        ]
    
    def _save_input_to_lods(self, x):
        self.lods.append(x)
        # print('---------->---------------->saving x : ', x.shape)
        return x


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
        # print('-------->custom conv2d input shape :', self.name, inputs.shape, self.kernel_size, self.filters)
        kernel_shape = [self.kernel_size] + [inputs.shape[3], self.filters],
        init_scale, post_scale = get_kernel_scales(
            kernel_shape=kernel_shape,
            he_initializer_slope=self.he_initializer_slope,
            use_weight_scaling=self.use_weight_scaling
        )
        
        kernel_initializer = tf.random_normal_initializer(stddev=init_scale)
        # print('---x shape : ', self.name, inputs.shape, post_scale, self.filters, self.kernel_size, self.strides, self.padding)
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
    def __init__(self, function, **kwargs):
        super(CustomLambda, self).__init__(function, **kwargs)
        self.lambda_fn = function
    
    def call(self, inputs):
        # print(f"Lambda [{self.name}] input shape : ", inputs.shape)
        return self.lambda_fn(inputs)