from typing import List
import numpy as np

from src.shared.model.model_meta import ModelMeta
from src.shared.model.model_data import ModelData
import src.shared.utils as utils


class FedAvgDualAggregator():
    def __init__(self):
        pass

    def aggregate(self, base_model_data: ModelData, new_model_data: ModelData, new_delta_model_meta: ModelMeta) -> ModelData:

        if new_model_data.num_round - base_model_data.num_round == 1:
            return new_model_data
        
        num_samples_learned_sum = base_model_data.num_samples_learned + new_model_data.num_samples_learned
        base_model_data_weights = base_model_data._get_weigths_np()
        new_model_data_weights = new_model_data._get_weigths_np()


        # Initialize the aggregated weights with zeros (empty model as base)
        aggregated_weights = utils.get_model().get_weights()
        
        # Weighted sum of all the weights
        for i in range(len(aggregated_weights)):
            aggregated_weights[i] = base_model_data_weights[i] * (base_model_data.num_samples_learned / (num_samples_learned_sum * 1.)) + new_model_data_weights[i] * (new_model_data.num_samples_learned / (num_samples_learned_sum * 1.))

        # Aggregate metadata
        model_meta = ModelMeta(
            num_samples_learned=base_model_data.num_samples_learned + new_delta_model_meta.num_samples_learned,
            num_epochs_learned=base_model_data.num_epochs_learned + new_delta_model_meta.num_epochs_learned,
            num_round=base_model_data.num_round + new_delta_model_meta.num_round,
            num_samples_epochs_learned=base_model_data.num_samples_epochs_learned + new_delta_model_meta.num_samples_epochs_learned,
            learned_dates=[]
        )
        
        # Create a new ModelData instance with averaged weights
        return ModelData(model_meta, aggregated_weights)