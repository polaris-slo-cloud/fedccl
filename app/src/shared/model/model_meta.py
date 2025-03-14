import json

class ModelMeta:
    num_samples_learned: int = 0
    num_epochs_learned: int = 0
    num_samples_epochs_learned: int = 0
    num_round: int = 0
    learned_dates: list = []


    def __init__(self, num_samples_learned: int = 0, num_epochs_learned: int = 0, num_round: int = 0, num_samples_epochs_learned: int = 0, learned_dates: list = []):
        self.num_samples_learned = num_samples_learned
        self.num_epochs_learned = num_epochs_learned
        self.num_round = num_round
        self.num_samples_epochs_learned = num_samples_epochs_learned
        self.learned_dates = learned_dates

    def to_json(self):
        return json.loads(json.dumps(self.__dict__))

    @staticmethod
    def from_json(data):
        # Create a ModelMeta object from the remaining data
        model_meta = ModelMeta(
            num_samples_learned=data['num_samples_learned'],
            num_epochs_learned=data['num_epochs_learned'],
            num_round=data['num_round'],
            num_samples_epochs_learned=data['num_samples_epochs_learned'],
            learned_dates=data['learned_dates']
        )
        
        return model_meta