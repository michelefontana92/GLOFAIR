import os

from pandas import DataFrame
from .dataset_factory import register_dataset
from .base_dataset import BaseDataset

@register_dataset('credit')
class CreditDataset(BaseDataset):

    def __init__(self,**kwargs):
        super(CreditDataset, self).__init__(**kwargs)
        self.root = kwargs.get('root', 'data/Credit')
        data_name = kwargs['filename']

        self.data_path = os.path.join(self.root, data_name)
        
        self.scaler_name = kwargs.get('scaler_name', 
                                      'credit_scalers.p')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                                [{}])
       
        self.scaler_path = f'{self.root}/{self.scaler_name}'
        
        self.target = 'default.payment.next.month'
        self.cat_cols = [
            'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2',
            'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'SEX', 'AGE'
        ]
        self.num_cols = [
            'LIMIT_BAL',
            'BILL_AMT1', 'BILL_AMT2',
            'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1',
            'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4',
            'PAY_AMT5', 'PAY_AMT6'
        ]
        self.labels = [0,1]
        self.clean_data_path = os.path.join(self.root,'fake_credit_age.csv')
        self.setup()
    
   
        
        
        
        
    

    