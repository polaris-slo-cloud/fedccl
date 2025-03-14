import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
from datetime import timedelta
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Define the base path for data
BASE_PATH = "../../data/shared_data"
RESULT_FILE = "analysis_results.csv"

# Load centralized data
df_prediction_power = pd.read_csv(f'{BASE_PATH}/centralized/prediction/all/prediction_all_power.csv')
df_prediction_acc_energy = pd.read_csv(f'{BASE_PATH}/centralized/prediction/all/prediction_all_acc_energy.csv')

df_prediction_power['time'] = pd.to_datetime(df_prediction_power['time'])
df_prediction_power['index'] = df_prediction_power['time']
df_prediction_power.set_index('index', inplace=True)
df_prediction_power = df_prediction_power.sort_values(by='time')

df_prediction_acc_energy['time'] = pd.to_datetime(df_prediction_acc_energy['time'])
df_prediction_acc_energy['index'] = df_prediction_acc_energy['time']
df_prediction_acc_energy.set_index('index', inplace=True)
df_prediction_acc_energy = df_prediction_acc_energy.sort_values(by='time')

df_prediction_power_centralized_all = df_prediction_power
df_prediction_acc_energy_centralized_all = df_prediction_acc_energy

df_prediction_power = pd.read_csv(f'{BASE_PATH}/centralized/prediction/continual/prediction_all_power.csv')
df_prediction_acc_energy = pd.read_csv(f'{BASE_PATH}/centralized/prediction/continual/prediction_all_acc_energy.csv')

df_prediction_power['time'] = pd.to_datetime(df_prediction_power['time'])
df_prediction_power['index'] = df_prediction_power['time']
df_prediction_power.set_index('index', inplace=True)
df_prediction_power = df_prediction_power.sort_values(by='time')

df_prediction_acc_energy['time'] = pd.to_datetime(df_prediction_acc_energy['time'])
df_prediction_acc_energy['index'] = df_prediction_acc_energy['time']
df_prediction_acc_energy.set_index('index', inplace=True)
df_prediction_acc_energy = df_prediction_acc_energy.sort_values(by='time')

df_prediction_power_centralized_continual = df_prediction_power
df_prediction_acc_energy_centralized_continual = df_prediction_acc_energy

# Load federated data
df_prediction_power = pd.read_csv(f'{BASE_PATH}/federated/prediction/prediction_all_power.csv')
df_prediction_acc_energy = pd.read_csv(f'{BASE_PATH}/federated/prediction/prediction_all_acc_energy.csv')

df_prediction_power['time'] = pd.to_datetime(df_prediction_power['time'])
df_prediction_power['index'] = df_prediction_power['time']
df_prediction_power.set_index('index', inplace=True)
df_prediction_power = df_prediction_power.sort_values(by='time')

df_prediction_acc_energy['time'] = pd.to_datetime(df_prediction_acc_energy['time'])
df_prediction_acc_energy['index'] = df_prediction_acc_energy['time']
df_prediction_acc_energy.set_index('index', inplace=True)
df_prediction_acc_energy = df_prediction_acc_energy.sort_values(by='time')

df_prediction_power_federated = df_prediction_power
df_prediction_acc_energy_federated = df_prediction_acc_energy

# Function to calculate errors
def show_error(data, key, mult, desc):
    data_day = data[(data['time'].dt.time >= pd.to_datetime('06:00').time()) &
                    (data['time'].dt.time <= pd.to_datetime('21:00').time())]
    
    error = ((abs((data[key] - data['actual'])) / (data['kwp'] * mult)) * 100)
    error_day = (((abs(data_day[key] - data_day['actual'])) / (data_day['kwp'] * mult)) * 100)

    mean_error = error.mean()
    mean_error_day = error_day.mean()

    median_error = error.median()
    median_error_day = error_day.median()

    max_error = error.max()

    # Create a DataFrame for the current error results
    df_error = pd.DataFrame({
        f'mean_error_day_{desc}': [mean_error_day],
        f'median_error_day_{desc}': [median_error_day],
        f'mean_error_{desc}': [mean_error],
        f'median_error_{desc}': [median_error],
        f'max_error_{desc}': [max_error]
    })

    return df_error

# Initialize a DataFrame to store results
df_results = pd.DataFrame()

# Centralized analysis for 'local'
for key in ['local']:
    key = f'predicted_{key}'
    df_results = pd.concat([
        df_results,
        show_error(df_prediction_power_centralized_all, key, 1000, "power_centralized_all"),
        show_error(df_prediction_acc_energy_centralized_all, key, 1000 * 12, "acc_energy_centralized_all")
    ], axis=1)

for key in ['local']:
    key = f'predicted_{key}'
    df_results = pd.concat([
        df_results,
        show_error(df_prediction_power_centralized_continual, key, 1000, "power_centralized_continual"),
        show_error(df_prediction_acc_energy_centralized_continual, key, 1000 * 12, "acc_energy_centralized_continual")
    ], axis=1)

# Federated analysis for 'global', 'cluster_location', 'cluster_orientation', 'local'
for key in ['global', 'cluster_location', 'cluster_orientation', 'local']:
    pred_key = f'predicted_{key}'
    df_results = pd.concat([
        df_results,
        show_error(df_prediction_power_federated, pred_key, 1000, f"power_federated_{key}"),
        show_error(df_prediction_acc_energy_federated, pred_key, 1000 * 12, f"acc_energy_federated_{key}")
    ], axis=1)

# Write results to CSV, appending if file exists
df_results.to_csv(RESULT_FILE, mode='a', header=not os.path.exists(RESULT_FILE), index=True)
