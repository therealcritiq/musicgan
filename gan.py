# import tensorflow_gan as tfgan
from random import random
import tensorflow as tf
from tensorflow import keras
from discriminator import Discriminator
from generator import Generator
from image_utils import load_audio_to_mel_spec
import utils
import hparams

class GAN(keras.Model):
    def __init__(self, latent_dim, generator, discriminator) -> None:
        super().__init__()
        # self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.generator = generator
        self.discriminator = discriminator

        # g_fn = lambda x: generator(x, **config)
        # self.model = tfgan.gan_model(
        #     generator_fn=lambda inputs: g_fn(inputs)[0],
        #     discriminator_fn=lambda images, unused_cond: d_fn(images)[0],
        #     real_data=real_images,
        #     generator_inputs=(noises, gen_one_hot_labels)
        # )
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
    
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        # real_images = load_audio_to_mel_spec()
        # real_images = utils.blend_images(real_images, 6, num_blocks=7)

        # TODO: need to generator one hot pitch labels and append to 
        # generator_input like in:
        # https://github.com/magenta/magenta/blob/77ed668af96edea7c993d38973b9da342bd31e82/magenta/models/gansynth/lib/model.py#L262
        # 

        pitch_one_hot_labels = utils.get_pitch_one_hot_labels(batch_size)
        noise = utils.sample_random_noise_vector(batch_size, self.latent_dim)
        generator_input = tf.concat((noise, pitch_one_hot_labels), axis=1)
        generated_images = self.generator.model(generator_input)
        combined_images = tf.concat([ generated_images, real_images ])
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros(batch_size, 1)],
            axis = 0
        )
        labels += 0.05 * tf.random.uniform(tf.shape(labels))


        with tf.GradientTape() as tape:
            predictions = self.discriminator.model(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.model.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.model.trainable_weights)
        )

        random_latent_vectors = utils.sample_random_noise_vector(batch_size, self.latent_dim)
        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator.model(
                self.generator.model(random_latent_vectors)
            )
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.model.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.model.trainable_weights)
        )

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result()
        }