import math
import numpy as np

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

BATCH_SIZE =32
EPOCHS = 20

def get_mixture_coef(y_pred, gaussian_mixtures, z_dim):
    
    d = gaussian_mixtures * z_dim
    
    rollout_length = K.shape(y_pred)[1]
    
    pi = y_pred[:,:,:d]
    mu = y_pred[:,:,d:(2*d)]
    log_sigma = y_pred[:,:,(2*d):(3*d)]
    #discrete = y_pred[:,3*gaussian_mixtures:]
    
    pi = K.reshape(pi, [-1, rollout_length, gaussian_mixtures, z_dim])
    mu = K.reshape(mu, [-1, rollout_length, gaussian_mixtures, z_dim])
    log_sigma = K.reshape(log_sigma, [-1, rollout_length, gaussian_mixtures, z_dim])

    pi = K.exp(pi) / K.sum(K.exp(pi), axis=2, keepdims=True)
    sigma = K.exp(log_sigma)
    
    return pi, mu, sigma#, discrete


def tf_normal(y_true, mu, sigma, pi, gaussian_mixtures, z_dim):

    rollout_length = K.shape(y_true)[1]
    y_true = K.tile(y_true,(1,1,gaussian_mixtures))
    y_true = K.reshape(y_true, [-1, rollout_length, gaussian_mixtures,z_dim])

    oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
    result = y_true - mu
#   result = K.permute_dimensions(result, [2,1,0])
    result = result * (1 / (sigma + 1e-8))
    result = -K.square(result)/2
    result = K.exp(result) * (1/(sigma + 1e-8))*oneDivSqrtTwoPI
    result = result * pi
    result = K.sum(result, axis=2) #### sum over gaussians
    #result = K.prod(result, axis=2) #### multiply over latent dims
    return result


class RNN():
    def __init__(self, z_dim, action_dim, hidden_units=256, gaussian_mixtures=5):
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.gaussian_mixtures = gaussian_mixtures

        self.models = self._build()
        self.model = self.models[0]
        self.model.summary()
        self.forward = self.models[1]

    def _build(self):

        #### THE MODEL THAT WILL BE TRAINED
        rnn_x = Input(shape=(None, self.z_dim + self.action_dim))
        lstm = LSTM(self.hidden_units, return_sequences=True, return_state = True)

        lstm_output, _ , _ = lstm(rnn_x)
        mdn = Dense(self.gaussian_mixtures * (3*self.z_dim))(lstm_output) #+ discrete_dim

        rnn = Model(rnn_x, mdn)

        #### THE MODEL USED DURING PREDICTION
        state_input_h = Input(shape=(self.hidden_units,))
        state_input_c = Input(shape=(self.hidden_units,))
        state_inputs = [state_input_h, state_input_c]
        _ , state_h, state_c = lstm(rnn_x, initial_state = [state_input_h, state_input_c])

        forward = Model([rnn_x] + state_inputs, [state_h, state_c])

        #### LOSS FUNCTION

        def make_rnn_r_loss( gmix, z_dim ):
            def rnn_r_loss(y_true, y_pred):

                pi, mu, sigma = get_mixture_coef(y_pred, gmix, z_dim)
            
                result = tf_normal(y_true, mu, sigma, pi, gmix, z_dim)
                
                result = -K.log(result + 1e-8)
                result = K.mean(result, axis = (1,2)) # mean over rollout length and z dim

                return result
            return rnn_r_loss
        r_loss = make_rnn_r_loss( self.gaussian_mixtures, self.z_dim )

        def make_rnn_kl_loss( gmix, z_dim ):
            def rnn_kl_loss(y_true, y_pred):
                pi, mu, sigma = get_mixture_coef(y_pred, gmix, z_dim)
                kl_loss = - 0.5 * K.mean(1 + K.log(K.square(sigma)) - K.square(mu) - K.square(sigma), axis = [1,2,3])
                return kl_loss
            return rnn_kl_loss
        kl_loss = make_rnn_kl_loss( self.gaussian_mixtures, self.z_dim )

        def rnn_loss(y_true, y_pred):
            return r_loss(y_true, y_pred) #+ rnn_kl_loss(y_true, y_pred)


        rnn.compile(loss=rnn_loss, optimizer='rmsprop', metrics = [r_loss, kl_loss])

        return (rnn,forward)

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, rnn_input, rnn_output, validation_split = 0.2):

        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        callbacks_list = [earlystop]

        self.model.fit(rnn_input, rnn_output,
            shuffle=True,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=validation_split,
            callbacks=callbacks_list)

        self.model.save_weights('./rnn/weights.h5')

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

