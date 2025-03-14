from src.shared.model.model_meta import ModelMeta
import json
import numpy as np

import src.shared.utils as utils

class ModelData(ModelMeta):
    weights: list = []

    def __init__(self, model_meta: ModelMeta, weights: list):
        super().__init__(model_meta.num_samples_learned, model_meta.num_epochs_learned, model_meta.num_round, model_meta.num_samples_epochs_learned, model_meta.learned_dates)
        
        # iterate over weights and convert to list
        self.weights = [w.tolist() for w in weights]
        
    def to_json(self):
        return json.loads(json.dumps(self.__dict__))
    
    @staticmethod
    def from_json(data):
        # Convert weights back to numpy arrays
        weights = [np.array(w) for w in data['weights']]
        
        # Create a ModelMeta object from the remaining data
        model_meta = ModelMeta.from_json(data)
        
        return ModelData(model_meta, weights)


    def _get_weigths_np(self):
        return utils.get_weigths_np(self.weights)