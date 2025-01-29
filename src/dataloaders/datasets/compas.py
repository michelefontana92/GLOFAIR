import os
from .dataset_factory import register_dataset
from .base_dataset import BaseDataset

@register_dataset('compas')
class CompasDataset(BaseDataset):
    """
    CompasDataset class for loading and processing the COMPAS dataset.
    Attributes:
        root (str): Root directory for the dataset.
        data_path (str): Path to the dataset file.
        scaler_name (str): Name of the scaler file.
        sensitive_attributes (list): List of sensitive attributes.
        scaler_path (str): Path to the scaler file.
        target (str): Target variable name.
        cat_cols (list): List of categorical columns.
        num_cols (list): List of numerical columns.
        labels (list): List of possible labels.
        clean_data_path (str): Path to the cleaned dataset file.
    Methods:
        __init__(**kwargs): Initializes the CompasDataset with the given parameters.
    """
    def __init__(self,**kwargs):
        super(CompasDataset, self).__init__(**kwargs)
        self.root = kwargs.get('root', 'data/Compas')
        data_name = kwargs['filename']

        self.data_path = os.path.join(self.root, data_name)
        
        self.scaler_name = kwargs.get('scaler_name', 
                                      'compas_scalers.p')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                                [{}])
       
        self.scaler_path = f'{self.root}/{self.scaler_name}'
        
        self.target = 'two_year_recid'
        self.cat_cols = [
            'c_charge_degree',
            'age_cat',
            'score_text',
            'decile_score',
            'sex',
            'race'
        ]
        self.num_cols = [
            'age',
            'priors_count',
            'days_b_screening_arrest',
            'length_of_stay',
            'juv_fel_count',
            'juv_misd_count',
            'juv_other_count',
            'c_days_from_compas'
        ]
        self.labels = [0,1]
        self.clean_data_path = os.path.join(self.root,'fake_compas.csv')
        self.setup()
        
        
        
        
    

    