# import tensorflow_gan as tfgan
import tensorflow as tf
from tensorflow import keras
from image_utils import load_audio_to_mel_spec
import utils
import hparams

class GAN(keras.Model):
    def __init__(self, latent_dim, generator, discriminator) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.generator = generator
        self.discriminator = discriminator

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
    
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_vector_size=hparams.latent_vector_size) -> None:
        self.num_img = num_img
        self.latent_vector_size = latent_vector_size

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = utils.sample_random_noise_vector(self.num_img, self.latent_vector_size)
        generated_images = self.model.generator.model(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()

        for i in range(self.num_img):
            img = keras.utils.array_to_img(generated_images[i])
            img.save(f'generated_mel_spec_{epoch:03d}_{i}.png')