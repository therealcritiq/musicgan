import tensorflow as tf
from tensorflow import keras
from keras import layers
import tf_slim
from base_model import BaseModel, CustomLambda
from utils import get_final_resolutions, get_num_filters, upscale_input
from activations import pixel_norm
import hparams

class Generator(BaseModel):
    def __init__(self, latent_vector_size=hparams.latent_vector_size, num_blocks=hparams.num_resolutions) -> None:
        super(Generator, self).__init__(save_input_to_lods=True)
        self.lods = []
        self.num_blocks = num_blocks
        # self.z = tf.concat(inputs, axis=1)
        # self.initial_x = tf_slim.flatten(self.z)
        # self.batch_size = int(self.z.shape[0])
        self.latent_vector_size = latent_vector_size
        self.model = self._build_model()
    
    def _build_model(self):
        """
        TODO: figure out if we need to use final_h and final_w (prob not though...)
        """
        # final_h, final_w = get_final_resolutions()
        model = keras.Sequential([
            # TODO: need to fix this input shape to make sure it includes the num of filters/channels
            # although it might be ok since the model summary is looking fine
            keras.Input(shape=(self.latent_vector_size, )), 
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
            #
            # trying something diff here
            # self._conv2d(kernel_size=(2, 16), filters=256, padding="VALID", name="conv-0")[0],
            # self._conv2d(kernel_size=(3, 3), filters=256, padding="VALID",  name="conv-1")[0],
            # self._upsample(block_id=0, name=f"UPSAMPLE_CONV_BLOCK_{0}-upsample"),
            # kernel_size was previously just hparams.start_resolutions based off of what the GANSynth Github code seemed to show
            # but the output of each layer using (4, 4) doesn't give us the same output architecture as defined in the
            # GANSynth paper
            # The following kernel_size of (6, 2) does
            *self._conv2d(kernel_size=(6, 2), filters=get_num_filters(1), padding="VALID", name="conv2d_1"),
            *self._conv2d(filters=get_num_filters(1)),
            *self._compose_upsample_conv2d_blocks(num_blocks=self.num_blocks),
            self._conv2d(kernel_size=(1, 1), filters=2, name="conv_final", activation=tf.nn.tanh)[0]
        ])
        return model
    