# Data
values_per_day = 96
training_days = 7

# Hyperparameters
lstm_units = 30  # Number of LSTM units
sequence_length = values_per_day #* training_days # Sequence length (representing one day)
batch_size = values_per_day #* training_days  # Batch size
num_epochs = 200  # Number of epochs
learning_rate = 0.0005  # Learning rate
loss_function = 'mean_squared_error'  # Loss function
kernel_regularization = 0.05  # Kernel regularization


# features
features = ['solar_rad_relative', 'ghi_relative', 'snow_depth_relative', 'precip_relative', 'clouds_relative', 'minute_of_day_relative', 'day_of_year_relative']
target = ['avg_relative']

