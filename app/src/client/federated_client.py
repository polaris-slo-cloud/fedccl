from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import LambdaCallback
from keras.layers import Input
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import os
import pickle
import logging
import sys

import src.shared.constants as constants
import src.shared.utils as utils
from src.shared.model.site import Site
from src.shared.model.model_data import ModelData
from src.shared.model.model_meta import ModelMeta
from src.shared.aggregation_level import AggregationLevel

class FederatedClient:
        
    def __init__(self, site: Site, value_key: str, data_path: str, server_url: str):
        self.site = site
        self.value_key = value_key
        self.base_path = data_path
        self.server_url = server_url

        self.logging_path = f'{self.base_path}/logs'
        self.models_path = f'{self.base_path}/models'
        self.predictions_path = f'{self.base_path}/predictions'

        os.makedirs(self.models_path, exist_ok=True)

        self._setup_logging()

        self.model_data = self._get_local_model()

        if self.model_data is None:
            model_data = ModelData(ModelMeta(), utils.get_model().get_weights())
            self._save_local_model(AggregationLevel.site, model_data)
            self.model_data = model_data
            
        self.is_registrered = False
        self._register_client()

    def _preprocess_data(self, data):
        if data is None or data.empty:
            logging.error(f"No data found for value_key: {self.value_key}. Exiting function.")
            return None

        if not any(data['value_key'] == self.value_key):
            logging.error(f"No values found with value_key: {self.value_key}. Exiting function.")
            return None

        default_columns = ['time', 'avg', 'solar_rad', 'precip', 'ghi', 'temp', 'snow_depth', 'clouds']
        time_shifted_columns = ['solar_rad_1h', 'precip_1h', 'ghi_1h', 'temp_1h', 'snow_depth_1h', 'solar_rad_2h', 'precip_2h', 'ghi_2h', 'temp_2h', 'snow_depth_2h', 'clouds_1h', 'clouds_2h', 'avg_24h','accumulated_energy_24h', 'whole_day_energy_24h']
        columns = default_columns + time_shifted_columns
        
        data = data[columns].copy()  # Ensure we're working with a copy

        # Adapt data based on conditions
        # data.loc[data['snow_depth_1h'] >= 50, ['solar_rad', 'ghi']] = 0

        # Compute and scale 'avg_relative', 'solar_rad', and 'ghi'
        data['avg_relative'] = data['avg'] / (float(self.site.kwp) * 1000)
        data['avg_24h_relative'] = data['avg_24h'] / (float(self.site.kwp) * 1000)

        # divide solar_rad by max (~1000 W/m^2) to get relative value
        data['solar_rad_relative'] = data['solar_rad'] / 1000
        data['solar_rad_1h_relative'] = data['solar_rad_1h'] / 1000
        data['solar_rad_2h_relative'] = data['solar_rad_2h'] / 1000

        # divide ghi by 1000 to get relative value
        data['ghi_relative'] = data['ghi'] / 1000
        data['ghi_1h_relative'] = data['ghi_1h'] / 1000
        data['ghi_2h_relative'] = data['ghi_2h'] / 1000

        # divide snow depth by 1000 to get relative value
        data['snow_depth_relative'] = data['snow_depth'] / 1200
        data['snow_depth_1h_relative'] = data['snow_depth_1h'] / 1200
        data['snow_depth_2h_relative'] = data['snow_depth_2h'] / 1200

        # divide precip by 15 to get relative value
        data['precip_relative'] = data['precip'] / 15
        data['precip_1h_relative'] = data['precip_1h'] / 15
        data['precip_2h_relative'] = data['precip_2h'] / 15

        # divide temp by 40 to get relative value
        data['temp_relative'] = data['temp'] / 40
        data['temp_1h_relative'] = data['temp_1h'] / 40
        data['temp_2h_relative'] = data['temp_2h'] / 40

        # add minute of the day
        data['minute_of_day'] = data['time'].dt.hour * 60 + data['time'].dt.minute
        data['minute_of_day_relative'] = data['minute_of_day'] / 1440

        # add day of year
        data['day_of_year'] = data['time'].dt.dayofyear
        data['day_of_year_relative'] = data['day_of_year'] / 365

        # clouds relative
        data['clouds_relative'] = data['clouds'] / 100
        data['clouds_1h_relative'] = data['clouds_1h'] / 100
        data['clouds_2h_relative'] = data['clouds_2h'] / 100

        # energy relative
        data['accumulated_energy_24h_relative'] = data['accumulated_energy_24h'] / (float(self.site.kwp) * 1000) * 24
        data['whole_day_energy_24h_relative'] = data['whole_day_energy_24h'] / (float(self.site.kwp) * 1000) * 24

        data = data.sort_values('time')
        
        # CHECKING CLOUDS
        # check if mean cloud coverage between 5 AM and 10 PM is higher than 50
        if data[(data['time'].dt.hour >= 5) & (data['time'].dt.hour <= 20)]['clouds_relative'].mean() > 0.5:
            logging.error(f"Cloud coverage for date {data['time'].dt.date.unique()} is too high. Exiting training function.")
            return None
        
        # drop rows which are defective
        for row in data.iterrows():
            # check if avg is negative or higher than kwp * 1000
            if row[1]['avg'] < 0 or row[1]['avg'] > (float(self.site.kwp) * 1000) or row[1]['avg_24h'] < 0 or row[1]['avg_24h'] > (float(self.site.kwp) * 1000):
                # delete row
                print(f"Deleting row with avg: {row[1]['avg']} or avg_24h: {row[1]['avg_24h']}")
                data = data.drop(row[0])

            # check if one of the constants.features is NaN
            if any(row[1][constants.features].isnull()):
                # delete row (check if exists)
                if row[0] in data.index:
                    data = data.drop(row[0])

        # check size
        if len(data) != constants.values_per_day * constants.training_days:
            logging.error(f"Data for date {utils.dates_to_daystrings(data['time'].dt.date.unique())} is incomplete. Exiting training function.")
            return None

        
        return data


    def _postprocess_data(self, data, predictions):
        if predictions is None or len(predictions) != constants.values_per_day * constants.training_days:
            logging.error(f"Data is incomplete. Exiting training function.")
            return None

        # predictions = self.scaler.inverse_transform(predictions_1d)  # Inverse transform predictions to original scale
        predictions = predictions.flatten()

        # multiply by kwp to get actual values
        predictions = predictions * (float(self.site.kwp) * 1000)

        # set values between between 5 AM and 8 PM to 0
        for i in range(len(data)):
            time = data['time'].iloc[i]
            if time.hour <= 5 or time.hour >= 20:
                predictions[i] = 0
        
         # set negative values to 0
        predictions[predictions < 0] = 0
        # shift 1h to the left
        predictions = np.roll(predictions, -1)
        
        # set to kwp * 1000 if higher than kwp * 1000
        predictions[predictions > (float(self.site.kwp) * 1000)] = float(self.site.kwp) * 1000
        return predictions

    def _train_model(self, level: AggregationLevel, model_data: ModelData, dates, data, cluster_key: str = None) -> ModelData:
        features = constants.features
        target = constants.target
        data_x = np.array(data[features])
        data_y = np.array(data[target])

        # Reshape for LSTM input
        data_x = data_x.reshape((data_x.shape[0], -1, len(features)))
        data_y = data_y.reshape((data_y.shape[0], len(target)))  # Target variable does not need additional dimensions

        model = utils.get_model()
        model.set_weights(model_data._get_weigths_np())

        # create data frame for epoch losses
        epoch_details = []

        model.fit(
            data_x[:-1], data_y[1:], 
            epochs=constants.num_epochs, 
            batch_size=constants.batch_size, 
            verbose=0,  # Suppress detailed output
            callbacks=[
                LambdaCallback(
                    on_epoch_end=lambda epoch, logs: [
                        logging.info(
                            f'Date: {dates[0]} - {dates[-1]}, Aggregation Level: {level} ({cluster_key}) Epoch: {epoch+1}, Loss: {format(logs["loss"], ".10f")}' if level == AggregationLevel.cluster else f'Date: {dates[0]} - {dates[-1]}, Aggregation Level: {level} Epoch: {epoch+1}, Loss: {format(logs["loss"], ".10f")}'
                        ),
                        epoch_details.append((epoch + 1, logs["loss"]))
                    ]
                )
            ]
        )

        model_meta = ModelMeta(
            num_samples_learned=model_data.num_samples_learned + len(data),
            num_epochs_learned=model_data.num_epochs_learned + constants.num_epochs,
            num_round=model_data.num_round + 1,
            num_samples_epochs_learned=model_data.num_samples_epochs_learned + len(data) * constants.num_epochs,
            learned_dates=model_data.learned_dates + [date.strftime('%Y-%m-%d') for date in dates]
        )

        model_delat_meta = ModelMeta(
            num_samples_learned=len(data),
            num_epochs_learned=constants.num_epochs,
            num_round=1,
            num_samples_epochs_learned=len(data) * constants.num_epochs,
            learned_dates=[date.strftime('%Y-%m-%d') for date in dates]
        )

        loss_df = pd.DataFrame(epoch_details, columns=['epoch', 'loss'])
        loss_df['round'] = model_meta.num_round
        loss_df['agg_level'] = level
        loss_df['date_begin'] = dates[0]
        loss_df['date_end'] = dates[-1]
        loss_df['site_id'] = self.site.site_id
        loss_df['time'] = loss_df['date_begin']

        if level == AggregationLevel.cluster:
            loss_df['cluster_key'] = cluster_key
        else: 
            loss_df['cluster_key'] = None

        # reorder columns
        loss_df = loss_df[['time', 'site_id', 'date_begin', 'date_end', 'agg_level', 'cluster_key', 'round', 'epoch', 'loss']]

        return ModelData(model_meta, model.get_weights()), model_delat_meta, loss_df

    def _predict(self, model_data: ModelData, data):
        if data is None or len(data) != constants.values_per_day * constants.training_days:
            logging.info(f"Data is incomplete. Exiting training function.")
            return None
        
        features = constants.features
        data_x = np.array(data[features])
        data_x = data_x.reshape((data_x.shape[0], -1, len(features)))

        model = utils.get_model()
        model.set_weights(model_data._get_weigths_np())

        # Predictions
        predictions = model.predict(data_x)

        # shift all values 2 to the left
        predictions = np.roll(predictions, -2, axis=0)
        
        return predictions

    def _compare_predictions(self, data, predictions_local_model, local_model: ModelData, predictions_cluster_location_model, cluster_location_model: ModelData, predictions_cluster_orientation_model, cluster_orientation_model: ModelData, predictions_global_model, global_model: ModelData):
        if not (predictions_local_model is None or predictions_global_model is None):
            avg_local_global = np.zeros(len(data))
            for i in range(len(data)):
                avg_local_global[i] = np.mean([predictions_local_model[i], predictions_global_model[i]])
        else:
            avg_local_global = [None] * len(data)
        
        
        if predictions_local_model is None:
            predictions_local_model = [None] * len(data)
        if predictions_cluster_location_model is None:
            predictions_cluster_location_model = [None] * len(data)
        if predictions_cluster_orientation_model is None:
            predictions_cluster_orientation_model = [None] * len(data)
        if predictions_global_model is None:
            predictions_global_model = [None] * len(data)
    
        pred_time = datetime.now()

        # Create DataFrame with predicted values
        df_power = pd.DataFrame({
            'site_id': self.site.site_id,
            'value_key': self.value_key,
            'pred_time': pred_time,
            'time': data['time'].values,
            'predicted_local': predictions_local_model,
            'predicted_cluster_location': predictions_cluster_location_model,
            'predicted_cluster_orientation': predictions_cluster_orientation_model,
            'predicted_global': predictions_global_model,
            'predicted_avg_local_global': avg_local_global,
            'cluster_location_model_round': cluster_location_model.num_round if cluster_location_model else None,
            'cluster_orientation_model_round': cluster_orientation_model.num_round if cluster_orientation_model else None,
            'local_model_round': local_model.num_round if local_model else None,
            'global_model_round': global_model.num_round if global_model else None,
            'actual': data['avg'].values,
            'ghi': data['ghi'].values,
            'solar_rad': data['solar_rad'].values,
            'temp': data['temp'].values,
            'precip': data['precip'].values,
            'snow_depth': data['snow_depth'].values,
            'clouds': data['clouds'].values,
            'kwp': self.site.kwp,
        })

        df_power['time'] = pd.to_datetime(df_power['time'])
        days = df_power['time'].dt.date.unique()

        df_energy = pd.DataFrame()
        for day in days :
            data_day = df_power[df_power['time'].dt.date == day]
            df_energy_day = pd.DataFrame({
                'site_id': self.site.site_id,
                'value_key': self.value_key,
                'pred_time': pred_time,
                'time': data_day['time'].values,
                'predicted_local': utils.accumulated_energy(data_day['predicted_local'].values),
                'predicted_cluster_location': utils.accumulated_energy(data_day['predicted_cluster_location'].values),
                'predicted_cluster_orientation': utils.accumulated_energy(data_day['predicted_cluster_orientation'].values),
                'predicted_global': utils.accumulated_energy(data_day['predicted_global'].values),
                'predicted_avg_local_global': utils.accumulated_energy(data_day['predicted_avg_local_global'].values),
                'cluster_location_model_round': cluster_location_model.num_round if cluster_location_model else None,
                'cluster_orientation_model_round': cluster_orientation_model.num_round if cluster_orientation_model else None,
                'local_model_round': local_model.num_round if local_model else None,
                'global_model_round': global_model.num_round if global_model else None,
                'actual': utils.accumulated_energy(data_day['actual'].values),
                'ghi': data_day['ghi'].values,
                'solar_rad': data_day['solar_rad'].values,
                'temp': data_day['temp'].values,
                'precip': data_day['precip'].values,
                'snow_depth': data_day['snow_depth'].values,
                'clouds': data_day['clouds'].values,
                'kwp': self.site.kwp,
            })

            df_energy = pd.concat([df_energy, df_energy_day], ignore_index=True)

        return df_power, df_energy

    def _register_client(self) -> bool:
        if self.is_registrered is True:
            return True

        if self.server_url is None:
            logging.error("No server URL provided. Exiting function.")
            return False
        
        try:
            logging.info(f"Registering client for site {self.site.site_id} on server {self.server_url}.")
            response = requests.post(f'{self.server_url}/site/{self.site.site_id}/register')
            if response.status_code != 200:
                raise 
        except:
            logging.error("Failed to register client. Exiting function.")
            self.is_registrered = False
            return False

        self.is_registrered = True
        return True
    
    def _save_local_model(self, lvl: AggregationLevel, model_data: ModelData, cluster_key: str = None):
        if lvl == AggregationLevel.cluster:
            model_data_path = f'{self.models_path}/{self.value_key}_{lvl}_{cluster_key}.pkl'
        else:
            model_data_path = f'{self.models_path}/{self.value_key}_{lvl}.pkl'

        with open(model_data_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def _get_local_model(self, lvl: AggregationLevel = 'site', cluster_key: str = None) -> ModelData:
        if lvl == AggregationLevel.cluster:
            model_data_path = f'{self.models_path}/{self.value_key}_{lvl}_{cluster_key}.pkl'
        else:
            model_data_path = f'{self.models_path}/{self.value_key}_{lvl}.pkl'
        model_data = None

        if os.path.exists(model_data_path):
             with open(model_data_path, 'rb') as f:
                model_data = pickle.load(f)
             
        return model_data
    
    def _get_server_model(self, lvl: AggregationLevel = 'site', cluster_key: str = None) -> ModelData:
        if self._register_client() is False:
            logging.error("Failed to register client. Exiting function.")
            return None
        
        try:
            if lvl == AggregationLevel.cluster:
                response = requests.get(f'{self.server_url}/site/{self.site.site_id}/model/{lvl}?cluster_key={cluster_key}')
            else: 
                response = requests.get(f'{self.server_url}/site/{self.site.site_id}/model/{lvl}')

            if response.status_code != 200:
                raise
            
            model_data = ModelData.from_json(response.json().get('model_data', None))
        except:
            logging.error(f"Failed to get {lvl}-model. Exiting function.")
            return None

        return model_data

    def _propagate_model(self, lvl: AggregationLevel = 'site', model_data: ModelData = None, model_delta_meta: ModelMeta = None, cluster_key: str = None):
        if self._register_client() is False:
            logging.error("Failed to register client. Exiting function.")
            return None
            
        try:
            if lvl == AggregationLevel.cluster:
                response = requests.post(f'{self.server_url}/site/{self.site.site_id}/model/{lvl}?cluster_key={cluster_key}', json={'model_data': model_data.to_json(), 'model_delta_meta': model_delta_meta.to_json()})
            else:
                response = requests.post(f'{self.server_url}/site/{self.site.site_id}/model/{lvl}', json={'model_data': model_data.to_json(), 'model_delta_meta': model_delta_meta.to_json()})
                
            if response.status_code != 200:
                raise Exception(f"Failed to propagate {lvl}-model for site {self.site.site_id}. Exiting function.")
        except:
            logging.error(f"Failed to propagate {lvl}-model for site {self.site.site_id}. Exiting function.")

    
    def _get_model_delta(self, model_data_old: ModelData, model_data_new: ModelData) -> ModelData:
        model_delta_meta = ModelMeta(
            num_samples_learned=model_data_new.num_samples_learned - model_data_old.num_samples_learned,
            num_epochs_learned=model_data_new.num_epochs_learned - model_data_old.num_epochs_learned,
            num_round=model_data_new.num_round - model_data_old.num_round,
            num_samples_epochs_learned=model_data_new.num_samples_epochs_learned - model_data_old.num_samples_epochs_learned,
            learned_dates=model_data_new.learned_dates
        )

        old_model_weigths = model_data_old._get_weigths_np()
        new_model_weigths = model_data_new._get_weigths_np()

        # calculate delta weights
        for i in range(len(old_model_weigths)):
                new_model_weigths[i] -= old_model_weigths[i]

        model_delta = ModelData(model_delta_meta, new_model_weigths)

        return model_delta

    def check_dates(self, dates):
        if not utils.are_subsequent(dates):
            logging.error(f"Data for dates {utils.dates_to_daystrings(dates)} is not subsequent. Exiting function.")
            return False

        if len(dates) != constants.training_days:
            logging.error(f"Data for dates {utils.dates_to_daystrings(dates)} is incomplete. Exiting function.")
            return False
        
        return True

    def _train_and_save_model(self, model_data, dates, data, level, cluster_key = None):
        if model_data is None:
            logging.error(f"No model data found for {level}. Exiting function.")
            return None, level

        updated_model_data, model_delta_meta, loss_df = self._train_model(level, model_data, dates, data, cluster_key)
        return updated_model_data, model_delta_meta, loss_df, level, cluster_key

    def process_data(self, data_arr):
        data = pd.DataFrame()
        dates = []

        for data_item in data_arr:
            if data_item is None or data_item.empty:
                logging.error(f"No data found for value_key.")
                continue

            item_dates = data_item['time'].dt.date.unique()
            item_dates.sort()

            if not self.check_dates(item_dates):
                continue
            
            if data_item is None or len(data_item) != constants.values_per_day * constants.training_days:
                logging.info(f"Data for date {utils.dates_to_daystrings(item_dates)} is incomplete. Exiting training function.")
                continue
            
            data_item = self._preprocess_data(data_item)

            if data_item is None:
                continue
            
            data = pd.concat([data, data_item], ignore_index=True)
            dates += item_dates.tolist()

        if data is None or data.empty:
            logging.error(f"No data found for value_key: {self.value_key}. Exiting function.")
            return None


        logging.info(f'Data complete and well shaped, start training ...')
        loss_all_df = pd.DataFrame()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for level in [AggregationLevel.site, AggregationLevel.cluster, AggregationLevel.global_]:
                if level == AggregationLevel.site:
                    model_data = self._get_local_model(level)
                    logging.info(f"Start training for {level} ...")
                    future = executor.submit(self._train_and_save_model, model_data, dates, data, level)
                    futures.append(future)
                elif level == AggregationLevel.cluster:
                    for cluster_key in self.site.clusters.keys():
                        model_data = self._get_server_model(level, cluster_key)
                        if model_data is None:
                            continue
                        logging.info(f"Start training for {level} ...")
                        future = executor.submit(self._train_and_save_model, model_data, dates, data, level, cluster_key)
                        futures.append(future)
                else:
                    model_data = self._get_server_model(level)
                    if model_data is None:
                        continue
                    logging.info(f"Start training for {level} ...")
                    future = executor.submit(self._train_and_save_model, model_data, dates, data, level)
                    futures.append(future)

            for future in futures:
                updated_model_data, model_delta_meta, loss_df, level, cluster_key = future.result()

                if updated_model_data is not None:
                    if level == AggregationLevel.cluster:
                        self._propagate_model(level, updated_model_data, model_delta_meta, cluster_key)
                    elif level == AggregationLevel.global_:
                        self._propagate_model(level, updated_model_data, model_delta_meta)
                    else:
                        self._save_local_model(level, updated_model_data)
                    
                    loss_all_df = pd.concat([loss_all_df, loss_df], ignore_index=True)

        loss_all_df.to_csv(f'{self.logging_path}/{self.value_key}_loss.csv', index=False)
        return loss_all_df
    
    def _predict_if_valid_model(self, lvl: AggregationLevel, data, model_data: ModelData = None):
        if model_data is None or model_data.num_samples_learned == 0:
            logging.error(f"No valid model data found for {lvl}. Exiting function.")
            return None

        predictions = self._predict(model_data, data)
        predictions = self._postprocess_data(data, predictions)

        return predictions
    
    def predict_data(self, data):
        if data is None or data.empty:
            logging.error(f"No data found for value_key: {self.value_key}. Exiting function.")
            return None, None

        dates = data['time'].dt.date.unique()
        dates.sort()

        if not self.check_dates(dates):
            return None, None
        
        data = self._preprocess_data(data)

        if data is None:
            logging.error(f"No data found for date {utils.dates_to_daystrings(dates)}. Exiting function.")
            return None, None

        logging.info(f'Data complete and well shaped, start predictions ...')


        site_model = self._get_local_model(AggregationLevel.site)
        cluster_location_model = self._get_server_model(AggregationLevel.cluster, 'location')
        cluster_orientation_model = self._get_server_model(AggregationLevel.cluster, 'orientation')
        global_model = self._get_server_model(AggregationLevel.global_)

        predictions = self._predict_if_valid_model(AggregationLevel.site, data, site_model)
        predictions_cluster_location_model = self._predict_if_valid_model(AggregationLevel.cluster, data, cluster_location_model)
        predictions_cluster_orientation_model = self._predict_if_valid_model(AggregationLevel.cluster, data, cluster_orientation_model)
        predictions_global_model = self._predict_if_valid_model(AggregationLevel.global_, data, global_model)
        
        return self._compare_predictions(data, predictions, site_model, predictions_cluster_location_model, cluster_location_model, predictions_cluster_orientation_model, cluster_orientation_model, predictions_global_model, global_model)

    def _setup_logging(self):
        os.makedirs(self.logging_path, exist_ok=True)
        log_file = f'{self.logging_path}/{self.value_key}.log'

        # Clear any existing log handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Set up logging to file and console
        logging.basicConfig(level=logging.INFO,
                            format=f'%(asctime)s - {self.site.site_id} - {self.value_key} - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_file),
                                logging.StreamHandler(sys.stdout)
                            ])