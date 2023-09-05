import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN


def create_simple_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(128, input_shape=input_shape))
    model.add(Dense(7))
    model.compile(loss='mae', optimizer=tf.keras.optimizers.RMSprop(), metrics=['accuracy'])
    return model


def create_simple_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape))
    model.add(Dense(7))
    model.compile(loss='mae', optimizer=tf.keras.optimizers.RMSprop(), metrics=['accuracy'])
    return model


def create_stacked_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(4, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(16, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(7))
    model.compile(loss='mae', optimizer=tf.keras.optimizers.RMSprop(), metrics=['accuracy'])
    return model
