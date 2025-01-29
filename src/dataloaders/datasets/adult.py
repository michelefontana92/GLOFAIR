import os
from .dataset_factory import register_dataset
from .base_dataset import BaseDataset

@register_dataset('adult')
class AdultDataset(BaseDataset):

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
        
        
        
        
    

    