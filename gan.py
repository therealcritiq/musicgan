import tensorflow_gan as tfgan
from generator import Generator

class GAN():
    def __init__(self) -> None:
        generator = Generator()
        # g_fn = lambda x: generator(x, **config)
        # self.model = tfgan.gan_model(
        #     generator_fn=lambda inputs: g_fn(inputs)[0],
        #     discriminator_fn=lambda images, unused_cond: d_fn(images)[0],
        #     real_data=real_images,
        #     generator_inputs=(noises, gen_one_hot_labels)
        # )