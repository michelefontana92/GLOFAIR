from ..base_run import BaseRun
from architectures import ArchitectureFactory

class CompasRun(BaseRun):
    
    def __init__(self,**kwargs):
        super(CompasRun, self).__init__(**kwargs)
        self.hidden1 = 300
        self.hidden2 = 100
        self.dropout = 0.2
        self.model = ArchitectureFactory.create_architecture('mlp2hidden',model_params={
                                                'input': 34,
                                                'hidden1': self.hidden1,
                                                'hidden2': self.hidden2,
                                                'dropout': self.dropout,
                                                'output': 2})
        self.dataset = 'compas'
        self.data_root  = 'data/Compas'
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                               [
                                                ('Race',
                                                    {'race':['African-American','Caucasian']}
                                                ),
                                                ('Gender',{'sex':['Male','Female']}),
                                                ('GenderRace',{
                                                    'race':['African-American','Caucasian'],
                                                    'sex':['Male','Female']
                                                })
                                                ])
        
        

    def setUp(self):
        pass
    def run(self):
        pass
    def tearDown(self):
        pass