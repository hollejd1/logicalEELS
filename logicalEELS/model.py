import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np



class VAE(keras.Model):
    '''Class: Variational Autoencoder for EELS Latent Representation using Keras'''
    def __init__(self, encoder, decoder, **kwargs):
        '''
        # Initialize logicalEELS, define encoder and decoder models and loss trackers
        Encoder     : Keras Model Class
        Decoder     : Keras Model Class
        '''
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        def _sample_z(inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        
        x = data[0]
        y = data[1]
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(x)
            z = layers.Lambda(_sample_z)([z_mean, z_log_var])
            reconstruction = self.decoder(z)
            
            reconstruction_loss = tf.reduce_sum(keras.losses.mae(y, reconstruction), axis=1)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(kl_loss, axis=1)
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(tf.reduce_mean(reconstruction_loss))
        self.kl_loss_tracker.update_state(tf.reduce_mean(kl_loss))
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        x = data[0]
        y = data[1]
        z_mean, z_log_var = self.encoder(x)
        reconstruction = self.decoder(z_mean)
        reconstruction_loss = tf.reduce_sum(keras.losses.mae(y, reconstruction), axis=1)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        return {
            "loss": total_loss,
            "reconstruction_loss": tf.reduce_mean(reconstruction_loss),
            "kl_loss": tf.reduce_mean(kl_loss),
        }

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        reconstruction = self.decoder(z_mean)
        reconstruction_loss = tf.reduce_sum(keras.losses.mae(inputs, reconstruction), axis=1)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_metric(tf.reduce_mean(kl_loss), name='kl_loss', aggregation='mean')
        self.add_metric(total_loss, name='total_loss', aggregation='mean')
        self.add_metric(tf.reduce_mean(reconstruction_loss), name='reconstruction_loss', aggregation='mean')
        return reconstruction
    
    def encode(self,inputs):
        """Encode inputs from spectra into latent space.
        
            Parameters:
                inputs:     Input tensor
            Returns:
                z_means:    Latent encodings
                Z_log_var:  Log Variance of the encodings"""
        z_means, z_log_var = self.encoder.predict(inputs)
        return z_means, z_log_var


    def decode(self,inputs):
        """Decode inputs from latent space into spectra.
        
            Parameters:
                inputs:     Input tensor
            Returns:
                outputs:    Reconstructed Spectra"""
        return self.decoder.predict(inputs)