import keras
from discriminator import Discriminator
from gan import GAN, GANMonitor
from generator import Generator
import hparams
from image_utils import load_audio_to_mel_spec

epochs = 100
generator = Generator()
discriminator = Discriminator()
gan = GAN(
    latent_dim=hparams.latent_vector_size,
    generator=generator,
    discriminator=discriminator
)
# print(gan.generator.model.summary())
# print(gan.discriminator.model.summary())
# gan.compile(
#     d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
#     g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
#     loss_fn=keras.losses.BinaryCrossentropy()
# )

# dataset = load_audio_to_mel_spec()
# gan.fit(
#     dataset, epochs=epochs, callbacks=[GANMonitor]
# )
