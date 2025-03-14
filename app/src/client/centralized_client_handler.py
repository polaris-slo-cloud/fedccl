import os
import pandas as pd
import fcntl
import json

from src.client.federated_client import FederatedClient
import src.shared.constants as constants
from src.shared.model.site import Site
from src.shared.model.model_data import ModelData

if __name__ == "__main__":
    data_path = os.getenv('DATA_PATH')
    test_data_path_origin = os.getenv('TEST_DATA_PATH')
    shared_data_path = os.getenv('SHARED_DATA_PATH')
    site_ids = os.getenv('SITE_IDS').split(',')
    value_key = os.getenv('VALUE_KEY')
    run_training = os.getenv('RUN_TRAINING').lower() in ('true', '1', 'True')
    run_prediction = os.getenv('RUN_PREDICTION').lower() in ('true', '1', 'True')

    server_url = None


    shared_data_path_all = os.path.join(shared_data_path, 'all')
    shared_data_path_continual = os.path.join(shared_data_path, 'continual')
    data_path_all = os.path.join(data_path, 'all')
    data_path_continual = os.path.join(data_path, 'continual')

    # create all paths (even if exist)
    os.makedirs(shared_data_path_all, exist_ok=True)
    os.makedirs(shared_data_path_continual, exist_ok=True)
    os.makedirs(data_path_all, exist_ok=True)
    os.makedirs(data_path_continual, exist_ok=True)

    for site_id in site_ids:
        test_data_path = os.path.join(test_data_path_origin, site_id)
        site_info = pd.read_csv(os.path.join(test_data_path, f'site_info.csv'), header=None).set_index(0).squeeze().to_dict()

        print(f"Processing site {site_id}")

        # check if data file exists
        if not os.path.exists(os.path.join(test_data_path, f'{value_key}.csv')):
            print(f"Data file {value_key}.csv does not exist")
            exit(1)

        data = pd.read_csv(os.path.join(test_data_path, f'{value_key}.csv'))
        training_windows = pd.read_csv(os.path.join(test_data_path, f'training_windows.csv'))
        test_windows = pd.read_csv(os.path.join(test_data_path, f'test_windows.csv'))
        
        data['time'] = pd.to_datetime(data['time'])

        training_windows = training_windows.applymap(lambda s: pd.to_datetime(s).date())
        test_windows = test_windows.applymap(lambda s: pd.to_datetime(s).date())

        site = Site(
            site_id=site_info['site_id'],
            clusters={},
            lat=float(site_info['lat']),
            lng=float(site_info['lng']),
            zip=int(site_info['zip']),
            country=site_info['country'],
            kwp=float(site_info['kwp']),
            weather_data=site_info['weather_data'].lower() == 'True'  # Convert string to boolean
        )


        client_all = FederatedClient(site, value_key, data_path_all, server_url)
        client_continual = FederatedClient(site, value_key, data_path_continual, server_url)

        if run_training:
            all_window_data = []
            for window in training_windows.values:
                window_data = None

                # Select data for the current window
                mask = data['time'].dt.date.isin(window)
                if mask.any():
                    print("Setting window data to training data")
                    window_data = data[mask]

                df_power, df_energy = client_continual.predict_data(window_data)

                if df_power is not None and df_energy is not None:
                    with open(f'{shared_data_path_continual}/training_all_power.csv', 'a') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        df_power.to_csv(f, header=f.tell()==0, index=False)
                        fcntl.flock(f, fcntl.LOCK_UN)

                    with open(f'{shared_data_path_continual}/training_all_acc_energy.csv', 'a') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        df_energy.to_csv(f, header=f.tell()==0, index=False)
                        fcntl.flock(f, fcntl.LOCK_UN)
                
                loss_df = client_continual.process_data([window_data])
                if loss_df is not None:
                    with open(f'{shared_data_path_continual}/loss_all.csv', 'a') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        loss_df.to_csv(f, header=f.tell()==0, index=False)
                        fcntl.flock(f, fcntl.LOCK_UN)

                all_window_data.append(window_data)

                df_power, df_energy = client_all.predict_data(window_data)

                if df_power is not None and df_energy is not None:
                    with open(f'{shared_data_path_all}/training_all_power.csv', 'a') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        df_power.to_csv(f, header=f.tell()==0, index=False)
                        fcntl.flock(f, fcntl.LOCK_UN)

                    with open(f'{shared_data_path_all}/training_all_acc_energy.csv', 'a') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        df_energy.to_csv(f, header=f.tell()==0, index=False)
                        fcntl.flock(f, fcntl.LOCK_UN)

            loss_df = client_all.process_data(all_window_data)
            if loss_df is not None:
                with open(f'{shared_data_path_all}/loss_all.csv', 'a') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    loss_df.to_csv(f, header=f.tell()==0, index=False)
                    fcntl.flock(f, fcntl.LOCK_UN)


        if run_prediction:
            for window in test_windows.values:
                window_data = None

                # Select data for the current window
                mask = data['time'].dt.date.isin(window)
                if mask.any():
                    print("Setting window data to test data")
                    window_data = data[mask]

                df_power, df_energy = client_all.predict_data(window_data)

                if df_power is not None and df_energy is not None:
                    with open(f'{shared_data_path_all}/prediction_all_power.csv', 'a') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        df_power.to_csv(f, header=f.tell()==0, index=False)
                        fcntl.flock(f, fcntl.LOCK_UN)

                    with open(f'{shared_data_path_all}/prediction_all_acc_energy.csv', 'a') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        df_energy.to_csv(f, header=f.tell()==0, index=False)
                        fcntl.flock(f, fcntl.LOCK_UN)

                df_power, df_energy = client_continual.predict_data(window_data)

                if df_power is not None and df_energy is not None:
                    with open(f'{shared_data_path_continual}/prediction_all_power.csv', 'a') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        df_power.to_csv(f, header=f.tell()==0, index=False)
                        fcntl.flock(f, fcntl.LOCK_UN)

                    with open(f'{shared_data_path_continual}/prediction_all_acc_energy.csv', 'a') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        df_energy.to_csv(f, header=f.tell()==0, index=False)
                        fcntl.flock(f, fcntl.LOCK_UN)
