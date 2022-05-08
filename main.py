from discriminator import Discriminator
from gan import GAN
from generator import Generator
import hparams

generator = Generator()
discriminator = Discriminator()
gan = GAN(
    latent_dim=hparams.latent_vector_size,
    generator=generator,
    discriminator=discriminator,
    batch_size=8
)
print(gan.generator.model.summary())
