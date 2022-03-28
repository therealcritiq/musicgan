# import tensorflow_gan as tfgan
import tensorflow as tf
from generator import Generator
import utils

class GAN():
    def __init__(self, batch_size) -> None:
        pitch_one_hot_labels = utils.get_pitch_one_hot_labels(batch_size)
        noise = utils.sample_random_noise_vector(batch_size)
        generator_input = tf.concat((noise, pitch_one_hot_labels), axis=1)
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