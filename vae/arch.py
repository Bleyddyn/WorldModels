import numpy as np

import tensorflow as tf
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, Cropping2D, Maximum
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping, TerminateOnNaN

CONV_FILTERS = [32,64,64,128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

CONV_T_FILTERS = [64,64,32,3]
CONV_T_KERNEL_SIZES = [5,5,6,6]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

Z_DIM = 32

EPOCHS = 1
BATCH_SIZE = 32

def sampling(args):
    z_mean, z_log_var = args
    z_dim = z_mean.shape[-1]
    if z_dim != Z_DIM:
        raise "Invalid dimensions: {} {}".format( z_dim, Z_DIM)
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Z_DIM), mean=0.0,stddev=1.0)
    return z_mean + K.exp(z_log_var / 2.0) * epsilon

# See: https://github.com/Lasagne/Recipes/blob/master/examples/variational_autoencoder/variational_autoencoder.py
# Use this as the activation for the vae_z_log_var layer
# Tried: 10
relu_shift = 1.0
def shifted_relu(x):
    return K.relu(x + relu_shift) - relu_shift

class VAE():
    def __init__(self, input_dim=(64,64,3), z_dim=32):
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.r_loss_const = 1.0
        self.kl_tolerance = 0.5 # For setting a maximum on the kl_loss
        self.models = self._build()
        self.model = self.models[0]
        self.encoder = self.models[1]
        self.decoder = self.models[2]

    def _build(self):

        vae_x = Input(shape=self.input_dim)
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0])(vae_x)
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0])(vae_c1)
        vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0])(vae_c2)
        vae_c4= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0])(vae_c3)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(self.z_dim)(vae_z_in)
        vae_z_log_var = Dense(self.z_dim, activation='relu')(vae_z_in)

        vae_z = Lambda(sampling)([vae_z_mean, vae_z_log_var])
        vae_z_input = Input(shape=(self.z_dim,))

        # we instantiate these layers separately so as to reuse them later
        vae_dense = Dense(1024)
        vae_dense_model = vae_dense(vae_z)

        vae_z_out = Reshape((1,1,DENSE_SIZE))
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0])
        vae_d1_model = vae_d1(vae_z_out_model)
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1])
        vae_d2_model = vae_d2(vae_d1_model)
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2])
        vae_d3_model = vae_d3(vae_d2_model)
        vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3])
        vae_d4_model = vae_d4(vae_d3_model)
        #crp4 = Cropping2D(1)
        #vae_d4_model = crp4(vae_d4(vae_d3_model))

        #### DECODER ONLY

        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d4_decoder = Cropping2D(1)(vae_d4(vae_d3_decoder))

        #### MODELS

        vae = Model(vae_x, vae_d4_model)
        vae_encoder = Model(vae_x, vae_z)
        vae_decoder = Model(vae_z_input, vae_d4_decoder)

        

        def vae_r_loss(y_true, y_pred):

            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)

            #return (64*64*3) * K.mean(K.square(y_true_flat - y_pred_flat), axis = -1)
            #return 0.5 * K.mean(K.square(y_true_flat - y_pred_flat), axis = -1)
            return self.r_loss_const * K.mean(K.square(y_true_flat - y_pred_flat), axis = -1)

        def vae_kl_loss(y_true, y_pred):
            #return - 0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis = -1)

            #kl_loss = -0.5 * tf.reduce_sum(1 + (vae_z_log_var) - K.square(vae_z_mean) - K.exp(vae_z_log_var), reduction_indices = 1)
            #kl_loss = K.mean( K.maximum(kl_loss, self.kl_tolerance * self.z_dim) )
            kl_loss = K.mean(1 + (vae_z_log_var) - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis = -1)
            kl_loss = -0.5 * K.clip(kl_loss, 0.0, self.kl_tolerance * self.z_dim)
            return kl_loss

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)
            
        optimizer = optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.005)
        vae.compile(optimizer=optimizer, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])

        return (vae,vae_encoder, vae_decoder)


    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, data, validation_split = 0.2, epochs=EPOCHS):

        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        callbacks_list = [earlystop, TerminateOnNaN()]

        self.model.fit(data, data,
                shuffle=True,
                epochs=epochs,
                batch_size=BATCH_SIZE,
                validation_split=validation_split,
                callbacks=callbacks_list)
        
        self.model.save_weights('./vae/weights.h5')

    def train_gen(self, generator, val_gen, epochs=EPOCHS):

        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        callbacks_list = [earlystop, TerminateOnNaN()]

        self.model.fit_generator(generator, epochs=epochs, callbacks=callbacks_list, validation_data=val_gen, shuffle=False)

        self.model.save_weights('./vae/weights.h5')

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def generate_rnn_data(self, obs_data, action_data):

        rnn_input = []
        rnn_output = []

        for i, j in zip(obs_data, action_data):    
            rnn_z_input = self.encoder.predict(np.array(i))
            conc = [np.concatenate([x,y]) for x, y in zip(rnn_z_input, j)]
            rnn_input.append(conc[:-1])
            rnn_output.append(np.array(rnn_z_input[1:]))

        rnn_input = np.array(rnn_input)
        rnn_output = np.array(rnn_output)

        return (rnn_input, rnn_output)
    

if __name__ == "__main__":
    vae = VAE(input_dim=(64,64,3))
    vae.model.summary()
