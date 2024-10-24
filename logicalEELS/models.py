import tensorflow as tf
from tensorflow import keras

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
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
        
        
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(x)
            z = keras.layers.Lambda(_sample_z)([z_mean, z_log_var])
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
        x, y = data
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
    
    def encode(self,inputs, means=False):
        z_means, z_log_var = self.encoder.predict(inputs)
        return z_means, z_log_var


    def decode(self,inputs):
        return self.decoder.predict(inputs)
    

def createVAE(params=None):

    if params==None:
        params={
            'INPUT_SHAPE'   :   (240, 1),
            'BATCH_SIZE'    :   32,
            'LATENT_SIZE'   :   16,
            'KERNEL_SIZES'  :   [7, 7, 3, 3],
            'FILTER_SIZES'  :   [16, 32, 32, 64],
            'ALPHA'         :   0.3,
            'DROPOUT'       :   0.2,
            'LR'            :   0.001,
        }

    encoderInput = keras.Input(params['INPUT_SHAPE'])
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][0], kernel_size=params['KERNEL_SIZES'][0], strides=2, padding='same')(encoderInput)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Dropout(params['DROPOUT'])(x)
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][1], kernel_size=params['KERNEL_SIZES'][1], strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Dropout(params['DROPOUT'])(x)
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][2], kernel_size=params['KERNEL_SIZES'][2], strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Dropout(params['DROPOUT'])(x)
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][3], kernel_size=params['KERNEL_SIZES'][3], strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Flatten()(x)
    z_mean = keras.layers.Dense(params['LATENT_SIZE'], activation='linear', name='z_mean')(x)
    z_log_var = keras.layers.Dense(params['LATENT_SIZE'], activation='linear', name='z_log_var')(x)


    decoderInput = keras.Input(params['LATENT_SIZE'])
    x = keras.layers.Dense(15*params['FILTER_SIZES'][-1], activation='linear')(decoderInput)
    x = keras.layers.Reshape((15, params['FILTER_SIZES'][-1]))(x)
    x = keras.layers.Conv1DTranspose(filters=params['FILTER_SIZES'][2], kernel_size=params['KERNEL_SIZES'][3], activation='relu', strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(filters=params['FILTER_SIZES'][1], kernel_size=params['KERNEL_SIZES'][2], activation='relu', strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(filters=params['FILTER_SIZES'][0], kernel_size=params['KERNEL_SIZES'][1], activation='relu', strides=2, padding='same')(x)
    decoderOutput = keras.layers.Conv1DTranspose(filters=1, kernel_size=params['KERNEL_SIZES'][0], activation='linear', strides=2, padding='same')(x)

    encoder = keras.Model(encoderInput, [z_mean, z_log_var], name='encoder')
    decoder = keras.Model(decoderInput, decoderOutput, name='decoder')

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(params['LR']))

    return vae


class dualVAE(keras.Model):
    def __init__(self, X_encoder, Y_encoder, decoder, loss_weights, **kwargs):
        super(dualVAE, self).__init__(**kwargs)
        self.X_encoder = X_encoder
        self.Y_encoder = Y_encoder
        self.decoder = decoder
        self.reconstruction_weight = loss_weights['RECON_WEIGHT']
        self.kl_weight = loss_weights['KL_WEIGHT']
        self.convergence_weight = loss_weights['CNVRG_WEIGHT']
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.convergence_loss_tracker = keras.metrics.Mean(name='convergence_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

        self.inputShape = self.X_encoder.layers[0].input_shape[0]
        self.latentDims = self.decoder.layers[0].input_shape[0]
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.convergence_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        def _sample_z(inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        x, y = data
        with tf.GradientTape() as tape:
            zy_mean, zy_log_var = self.Y_encoder(y)
            zx = self.X_encoder(x)
            zy = keras.layers.Lambda(_sample_z)([zy_mean, zy_log_var])
            X_reconstruction = self.decoder(zx)
            Y_reconstruction = self.decoder(zy)
            reconstruction_loss = tf.reduce_sum(keras.losses.mae(y, Y_reconstruction), axis=1)

            convergence_loss = tf.reduce_sum(keras.losses.mse(tf.expand_dims(zy_mean, -1), tf.expand_dims(zx, -1)), axis=1)

            kl_loss = -0.5 * (1 + zy_log_var - tf.square(zy_mean) - tf.exp(zy_log_var))
            kl_loss = tf.reduce_sum(kl_loss, axis=1)

            total_loss = tf.reduce_mean(reconstruction_loss*self.reconstruction_weight +
                                        kl_loss*self.kl_weight +
                                        convergence_loss*self.convergence_weight)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(tf.reduce_mean(reconstruction_loss))
        self.convergence_loss_tracker.update_state(tf.reduce_mean(convergence_loss))
        self.kl_loss_tracker.update_state(tf.reduce_mean(kl_loss))
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "convergence_loss": self.convergence_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        x = data[0]
        y = data[1]
        zy_mean, zy_log_var = self.Y_encoder(x)
        zx = self.X_encoder(x)
        X_reconstruction = self.decoder(zx)
        Y_reconstruction = self.decoder(zy_mean)
        reconstruction_loss = tf.reduce_sum(keras.losses.mae(y, Y_reconstruction), axis=1)

        convergence_loss = tf.reduce_sum(keras.losses.mse(tf.expand_dims(zy_mean, -1), tf.expand_dims(zx, -1)), axis=1)

        kl_loss = -0.5 * (1 + zy_log_var - tf.square(zy_mean) - tf.exp(zy_log_var))
        kl_loss = tf.reduce_mean(kl_loss, axis=1)

        total_loss = tf.reduce_mean(reconstruction_loss*self.reconstruction_weight +
                                        kl_loss*self.kl_weight +
                                        convergence_loss*self.convergence_weight)
        return {
            "loss": total_loss,
            "reconstruction_loss": tf.reduce_mean(reconstruction_loss*self.reconstruction_weight),
            "convergence_loss": tf.reduce_mean(convergence_loss*self.convergence_weight),
            "kl_loss": tf.reduce_mean(kl_loss*self.kl_weight),
        }

    def call(self, inputs):
        z_mean, z_log_var = self.Y_encoder(inputs)
        reconstruction = self.decoder(z_mean)
        reconstruction_loss = tf.reduce_mean(keras.losses.mae(inputs, reconstruction), axis=1)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(kl_loss, axis=1)
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_metric(tf.reduce_mean(kl_loss), name='kl_loss', aggregation='mean')
        self.add_metric(total_loss, name='total_loss', aggregation='mean')
        self.add_metric(tf.reduce_mean(reconstruction_loss), name='reconstruction_loss', aggregation='mean')
        return reconstruction

    def denoise(self, inputs):
        z = self.X_encoder.predict(inputs)
        return self.decoder.predict(z)
    
    def X_encode(self,inputs):
        z = self.X_encoder.predict(inputs)
        return z

    def Y_encode(self,inputs):
        z_means, z_log_var = self.Y_encoder.predict(inputs)
        return z_means, z_log_var

    def decode(self,inputs):
        return self.decoder.predict(inputs)


def createDualVAE(params=None):

    if params==None:
        params={
            'INPUT_SHAPE'   :   (240, 1),
            'BATCH_SIZE'    :   32,
            'LATENT_SIZE'   :   16,
            'KERNEL_SIZES'  :   [7, 7, 3, 3],
            'FILTER_SIZES'  :   [16, 32, 32, 64],
            'ALPHA'         :   0.3,
            'DROPOUT'       :   0.2,
            'LR'            :   0.001,
            'RECON_WEIGHT'  :   0.001,
            'KL_WEIGHT'     :   0.5,
            'CNVRG_WEIGHT'  :   1.0,

        }

    encoderInput = keras.Input(params['INPUT_SHAPE'])
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][0], kernel_size=params['KERNEL_SIZES'][0], strides=2, padding='same')(encoderInput)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Dropout(params['DROPOUT'])(x)
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][1], kernel_size=params['KERNEL_SIZES'][1], strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Dropout(params['DROPOUT'])(x)
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][2], kernel_size=params['KERNEL_SIZES'][2], strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Dropout(params['DROPOUT'])(x)
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][3], kernel_size=params['KERNEL_SIZES'][3], strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Flatten()(x)
    z = keras.layers.Dense(params['LATENT_SIZE'], activation='linear', name='z')(x)

    vencoderInput = keras.Input(params['INPUT_SHAPE'])
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][0], kernel_size=params['KERNEL_SIZES'][0], strides=2, padding='same')(vencoderInput)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Dropout(params['DROPOUT'])(x)
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][1], kernel_size=params['KERNEL_SIZES'][1], strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Dropout(params['DROPOUT'])(x)
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][2], kernel_size=params['KERNEL_SIZES'][2], strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Dropout(params['DROPOUT'])(x)
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][3], kernel_size=params['KERNEL_SIZES'][3], strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Flatten()(x)
    z_mean = keras.layers.Dense(params['LATENT_SIZE'], activation='linear', name='z_mean')(x)
    z_log_var = keras.layers.Dense(params['LATENT_SIZE'], activation='linear', name='z_log_var')(x)


    decoderInput = keras.Input(params['LATENT_SIZE'])
    x = keras.layers.Dense(15*params['FILTER_SIZES'][-1], activation='linear')(decoderInput)
    x = keras.layers.Reshape((15, params['FILTER_SIZES'][-1]))(x)
    x = keras.layers.Conv1DTranspose(filters=params['FILTER_SIZES'][2], kernel_size=params['KERNEL_SIZES'][3], activation='relu', strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(filters=params['FILTER_SIZES'][1], kernel_size=params['KERNEL_SIZES'][2], activation='relu', strides=2, padding='same')(x)
    x = keras.layers.Conv1DTranspose(filters=params['FILTER_SIZES'][0], kernel_size=params['KERNEL_SIZES'][1], activation='relu', strides=2, padding='same')(x)
    decoderOutput = keras.layers.Conv1DTranspose(filters=1, kernel_size=params['KERNEL_SIZES'][0], activation='linear', strides=2, padding='same')(x)

    vencoder = keras.Model(vencoderInput, [z_mean, z_log_var], name='variational_encoder')
    encoder = keras.Model(encoderInput, z, name='standard_encoder')
    decoder = keras.Model(decoderInput, decoderOutput, name='decoder')

    loss_weights = dict((k, params[k]) for k in ['RECON_WEIGHT', 'KL_WEIGHT', 'CNVRG_WEIGHT'])

    dvae = dualVAE(encoder, vencoder, decoder, loss_weights)
    dvae.compile(optimizer=keras.optimizers.Adam(params['LR']), loss='mae')

    return dvae