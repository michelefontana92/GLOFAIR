import torch
_DATASETS ={}

def register_dataset(dataset):
    def decorator(cls):
        if dataset in _DATASETS:
            raise ValueError(f"Cannot register duplicate dataset ({dataset})")
        if not issubclass(cls, torch.utils.data.Dataset):
            raise ValueError(f"Dataset ({dataset}: {cls.__name__}) must extend BaseDataset")
        _DATASETS[dataset] = cls
        return cls
    return decorator

class DatasetFactory:
    @staticmethod
    def create_dataset(dataset, **kwargs):
        if dataset not in _DATASETS:
            raise ValueError(f"Unknown dataset type: {dataset}")
        return _DATASETS[dataset](**kwargs)

    @staticmethod
    def serialize(dataset_instance):
        """
        Serializza un'istanza di dataset registrato.
        """
        dataset_type = type(dataset_instance).__name__
        # Usa il nome registrato, non il nome della classe
        dataset_name = next(
            (name for name, cls in _DATASETS.items() if isinstance(dataset_instance, cls)),
            None
        )
        if dataset_name is None:
            raise ValueError(f"Dataset {dataset_type} non registrato.")
        return {
            'type': dataset_name,
            'state': dataset_instance.__dict__  # Salva lo stato interno del dataset
        }

    @staticmethod
    def deserialize(data):
        """
        Ricostruisce un'istanza di dataset dai dati serializzati.
        """
        dataset_name = data['type']
        if dataset_name not in _DATASETS:
            raise ValueError(f"Dataset {dataset_name} non registrato.")
        dataset_cls = _DATASETS[dataset_name]
        instance = dataset_cls.__new__(dataset_cls)  # Crea l'istanza senza chiamare __init__
        instance.__dict__.update(data['state'])  # Ripristina lo stato interno
        return instance

