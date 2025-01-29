from dataloaders.base_loader import BaseDataLoader
from torch.utils.data import DataLoader
from .datasets import DatasetFactory

class DataModule(BaseDataLoader):
    """
    DataModule class for loading and managing datasets.
    Attributes:
        kwargs (dict): Keyword arguments for configuring the DataModule.
        dataset_name (str): Name of the dataset.
        root (str): Root directory for the dataset.
        train_set_name (str): Name of the training set.
        val_set_name (str): Name of the validation set.
        test_set_name (str): Name of the test set.
        batch_size (int): Batch size for data loading. Default is 128.
        num_workers (int): Number of workers for data loading. Default is 0.
        load_test_set (bool): Flag to indicate whether to load the test set. Default is False.
        datasets (dict): Dictionary containing the datasets.
    Methods:
        _load_data(): Loads the datasets based on the provided paths.
        train_loader(batch_size=None): Returns a DataLoader for the training set.
        val_loader(batch_size=None): Returns a DataLoader for the validation set.
        test_loader(batch_size=None): Returns a DataLoader for the test set.
        train_loader_eval(batch_size=None): Returns a DataLoader for evaluating the training set.
        get_input_dim(): Returns the input dimension of the training set.
        get_class_weights(): Returns the class weights of the training set.
        merge(datamodule_list): Merges the datasets from a list of DataModules.
        get_group_ids(): Returns the group IDs of the training set.
        get_group_cardinality(y, group_id, training_group_name): Returns the group cardinality of the training set.
        serialize(): Serializes the DataModule.
        deserialize(data): Reconstructs an instance of DataModule from serialized data.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
        self.dataset_name = kwargs.get('dataset')
        self.root = kwargs.get('root')
        self.train_set_name = kwargs.get('train_set')
        self.val_set_name = kwargs.get('val_set')
        self.test_set_name = kwargs.get('test_set')
        
       

        self.batch_size = kwargs.get('batch_size', 128)
        self.num_workers = kwargs.get('num_workers', 0)
        self.load_test_set = kwargs.get('load_test_set', False)
        self._load_data()
    
    def _load_data(self):
        _PATHS = {
            'train': self.train_set_name,
            'val': self.val_set_name,
        }
        
        self.datasets = {
            'train': None,
            'val': None,
        }

        if self.load_test_set:
            _PATHS['test'] = self.test_set_name
            self.datasets['test'] = None
            
        for key in _PATHS.keys():
            
            self.datasets[key] = DatasetFactory().create_dataset(
                filename=_PATHS[key],
                **self.kwargs
            )
       

    def train_loader(self,batch_size=None):
        return DataLoader(self.datasets.get('train'),
                          batch_size=self.batch_size if batch_size is None else batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_loader(self,batch_size=None):
       return DataLoader(self.datasets.get('val'),
                          batch_size=len(self.datasets.get('val')) if batch_size is None else batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_loader(self,batch_size=None):
        return DataLoader(self.datasets.get('test'),
                          batch_size=len(self.datasets.get('test')) if batch_size is None else batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def train_loader_eval(self,batch_size=None):
        return DataLoader(self.datasets.get('train'),
                          batch_size=len(self.datasets.get('train')) if batch_size is None else batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
    
    def get_input_dim(self):
        return self.datasets['train'].x.shape[1]
    
    def get_class_weights(self):
        return self.datasets['train'].get_class_weights()
    
    def merge(self, datamodule_list):
        for key in self.datasets.keys():
            for datamodule in datamodule_list:
                self.datasets[key].merge(datamodule.datasets[key])
        return self
    
    def get_group_ids(self):
        return self.datasets['train'].get_group_ids()
    
    def get_group_cardinality(self,y,group_id,training_group_name):
        return self.datasets['train'].get_group_cardinality(y,group_id,training_group_name)
    
    def serialize(self):
        """
        Serializza il DataModule.
        """
        return {
            'kwargs': self.kwargs,  # Argomenti usati per creare il DataModule
            'datasets': {
                key: DatasetFactory.serialize(dataset) if dataset else None
                for key, dataset in self.datasets.items()
            }
        }

    @staticmethod
    def deserialize(data):
        """
        Ricostruisce un'istanza di DataModule dai dati serializzati.
        """
        instance = DataModule(**data['kwargs'])
        instance.datasets = {
            key: DatasetFactory.deserialize(dataset_data) if dataset_data else None
            for key, dataset_data in data['datasets'].items()
        }
        return instance