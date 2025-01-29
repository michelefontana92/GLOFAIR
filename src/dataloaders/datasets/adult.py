import os
from .dataset_factory import register_dataset
from .base_dataset import BaseDataset

@register_dataset('adult')
class AdultDataset(BaseDataset):
    """
    AdultDataset class for loading and processing the Adult dataset.
    This class inherits from BaseDataset and is registered with the name 'adult'.
    It initializes the dataset with various parameters and sets up the necessary
    paths and attributes for data loading and processing.
    Attributes:
        root (str): Root directory for the dataset. Default is 'data/Adult'.
        data_path (str): Path to the dataset file.
        scaler_name (str): Name of the scaler file. Default is 'adult_scalers.p'.
        sensitive_attributes (list): List of sensitive attributes. Default is [{}].
        scaler_path (str): Path to the scaler file.
        target (str): Target variable for the dataset. Default is 'income'.
        cat_cols (list): List of categorical columns.
        num_cols (list): List of numerical columns.
        labels (list): List of labels for the target variable.
        clean_data_path (str): Path to the cleaned dataset file.
    Methods:
        __init__(**kwargs): Initializes the dataset with the given parameters.
        setup(): Sets up the dataset for loading and processing.
    """
    def __init__(self,**kwargs):
        super(AdultDataset, self).__init__(**kwargs)
        self.root = kwargs.get('root', 'data/Adult')
        data_name = kwargs['filename']

        self.data_path = os.path.join(self.root, data_name)
        
        self.scaler_name = kwargs.get('scaler_name', 
                                      'adult_scalers.p')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                                [{}])
       
        self.scaler_path = f'{self.root}/{self.scaler_name}'
        
        self.target = 'income'
        self.cat_cols = ['workclass', 'education', 'race', 'gender',
                    'marital-status', 'occupation', 'relationship', 'native-country']
        self.num_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
        self.labels = ['<=50K', '>50K']
        self.clean_data_path = os.path.join(self.root,'adult_clean.csv')
        self.setup()
        
        
        
        
    

    