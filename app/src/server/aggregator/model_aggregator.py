from typing import List
import numpy as np

from src.shared.model.model_data import ModelData
from src.shared.model.model_meta import ModelMeta

class ModelAggregator:
    def __init__(self):
        pass

    def aggregate(self, base_model_data: ModelData, model_data_list: List[ModelData]) -> ModelData:
        # Check if the list is empty
        if not model_data_list:
            raise ValueError("The model_data_list is empty")
        
    def _aggregate_metadata(self, model_data_list: List[ModelData]) -> ModelMeta:
        # Aggregate metadata
        num_samples_list = [md.num_samples_learned for md in model_data_list]
        num_epochs_list = [md.num_epochs_learned for md in model_data_list]
        num_round_list = [md.num_round for md in model_data_list]
        num_samples_epochs_learned_list = [md.num_samples_epochs_learned for md in model_data_list]
        learned_dates_list = [md.learned_dates for md in model_data_list]

        
        # Calculate the total number of samples
        total_samples = sum(num_samples_list)

        # Aggregate metadata
        aggregated_num_samples = total_samples
        aggregated_num_epochs = sum(num_epochs_list)
        aggregated_num_round = sum(num_round_list)
        num_samples_epochs_learned = sum(num_samples_epochs_learned_list)
        aggregated_learned_dates = list(set(date for sublist in learned_dates_list for date in sublist))
        
        # Create a new ModelMeta instance with the aggregated metadata
        aggregated_meta = ModelMeta(
            num_samples_learned=aggregated_num_samples,
            num_epochs_learned=aggregated_num_epochs,
            num_round=aggregated_num_round,
            num_samples_epochs_learned=num_samples_epochs_learned,
            learned_dates=aggregated_learned_dates
        )
        
        return aggregated_meta

