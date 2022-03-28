from gan import GAN

gan = GAN(batch_size=8)
print(gan.generator.model.summary())
