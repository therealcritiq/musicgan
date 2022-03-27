import tensorflow_gan as tfgan
import tensorflow as tf
from generator import Generator
import utils

class GAN():
    def __init__(self, batch_size) -> None:
        pitch_one_hot_labels = utils.get_pitch_one_hot_labels(batch_size)
        z = utils.generate_latent_vector_z(batch_size)
        generator_input = tf.concat(z, pitch_one_hot_labels)
        self.generator = Generator(generator_input)

        # g_fn = lambda x: generator(x, **config)
        # self.model = tfgan.gan_model(
        #     generator_fn=lambda inputs: g_fn(inputs)[0],
        #     discriminator_fn=lambda images, unused_cond: d_fn(images)[0],
        #     real_data=real_images,
        #     generator_inputs=(noises, gen_one_hot_labels)
        # )
    def train(self):
        pass