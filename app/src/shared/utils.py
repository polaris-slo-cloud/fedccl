from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

import pandas as pd
import numpy as np

import src.shared.constants as constants

# Define custom loss function
def mean_power_error(y_true, y_pred):
    return K.mean(K.pow(K.abs(y_true - y_pred), 3))

def get_model():
    optimizer = Adam(learning_rate=constants.learning_rate)  # Optimizer
    model = Sequential()
    model.add(Input(shape=(constants.sequence_length, len(constants.features))))
    model.add(LSTM(units=constants.lstm_units, return_sequences=False, kernel_regularizer=l2(constants.kernel_regularization)))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=constants.loss_function)  # Use the custom loss function here

    return model

def get_weigths_np(weights_list):
    return [np.array(w, dtype=np.float32) for w in weights_list]

def are_subsequent(dates):
    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days != 1:
            return False
    return True

def dates_to_daystrings(dates):
    return [date.strftime('%Y-%m-%d') for date in dates]

def get_cluster_id(key, value):
    return f"{key}_{value}"

def accumulated_energy(power_values):
    # check if any none values
    if any([x is None for x in power_values]):
        return [None] * len(power_values)
    
    power_values = np.array(power_values)

    # Calculate the energy for each interval (15 minutes => 0.25 hours)
    energy_per_interval = power_values * 0.25

    # Calculate the accumulated energy
    accumulated_energy = np.cumsum(energy_per_interval)

    return accumulated_energy