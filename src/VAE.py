from keras import backend as K
from keras import optimizers
from keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda,Dropout
from keras.losses import mean_squared_error,binary_crossentropy, kullback_leibler_divergence
from keras.models import Model
import numpy as np
from sklearn.preprocessing import Normalizer
import tensorflow as tf

class VAE():
    def __init__(self, args):
        self.args = args
        self_vae = None
        self.build_model()
        
    def build_model(self):
        # np.random.seed(42)
        # tf.random.set_seed(42)
        # Build the encoder network
        # ------------ Input -----------------
        s1_inp = Input(shape=(self.args.input_size,))
        s2_inp = Input(shape=(self.args.input_size,))
        s3_inp = Input(shape=(self.args.input_size,))
        inputs = [s1_inp, s2_inp, s3_inp]
        
        # ------------ Concat Layer -----------------
        
        x1 = Dense(self.args.input_size, activation="elu")(s1_inp)
        x1 = BN()(x1)

        x2 = Dense(self.args.input_size, activation="elu")(s2_inp)
        x2 = BN()(x2)
        x3 = Dense(self.args.input_size, activation="elu")(s3_inp)
        x3 = BN()(x3)
        x = Concatenate(axis=-1)([x1, x2, x3])

        x = Dense(self.args.dense_layer_size, activation="elu")(x)
        x = BN()(x)
        
        # ------------Embedding Layer --------------
        
        z_mean = Dense(self.args.latent_size, name='z_mean')(x)
        z_log_sigma = Dense(self.args.latent_size, name='z_log_sigma', kernel_initializer='zeros')(x)
        z = Lambda(self.sampling, output_shape=(self.args.latent_size,), name='z')([z_mean, z_log_sigma])
        self.encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
            
        # -------Build the decoder network------------------

        latent_inputs = Input(shape=(self.args.latent_size,), name='z_sampling')
        x = Dense(self.args.dense_layer_size, activation="elu")(latent_inputs)
        x = BN()(x)
        x = Dropout(self.args.dropout)(x)
        out = Dense(self.args.input_size, activation="elu")(x)
        out = BN()(out)
        out = Dense(self.args.input_size)(out)
        
        # ------------Final Out -----------------------

        decoder = Model(latent_inputs , [out], name='decoder')
        outputs = decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='TVAE')
        distance = self.kl_regu(z_mean,z_log_sigma)
        s1_loss= mean_squared_error(inputs[0], outputs)
        s2_loss = mean_squared_error(inputs[1], outputs)
        s3_loss = mean_squared_error(inputs[2], outputs)
        
        reconstruction_loss = K.exp(s1_loss + s2_loss - s3_loss)
        vae_loss = K.mean(reconstruction_loss + self.args.beta * distance)
        self.vae.add_loss(vae_loss)
        encoder_metric = tf.keras.metrics.MeanSquaredError(name = 'encoder_metric')
        decoder_metric = tf.keras.metrics.MeanSquaredError(name = 'decoder_metric')
        adam = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.vae.compile(optimizer=adam, metrics=[encoder_metric, decoder_metric])
    
    def train(self, s1_train, s2_train, s3_train):
        self.vae.fit([s1_train, s2_train, s3_train], s1_train, epochs=self.args.epochs, batch_size=self.args.batch_size, verbose = 0, shuffle=True)
        if self.args.save_model:
            self.vae.save_weights("model_weights.h5")
            
    def predict(self, s1_data, s2_data, s3_data):
        return self.vae.predict([s1_data, s2_data, s3_data], batch_size=self.args.batch_size, verbose = 0)
        
    def kl_regu(self, z_mean,z_log_sigma):
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return kl_loss
    
    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon