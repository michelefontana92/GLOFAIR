from ..base_run import BaseRun
from architectures import ArchitectureFactory

class CompasRun(BaseRun):
    
    def __init__(self,**kwargs):
        super(CompasRun, self).__init__(**kwargs)
        self.input_size = 34
        self.hidden1 = 300
        self.hidden2 = 100
        self.dropout = 0.2
        self.dataset = 'compas'
       
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
        
        self.init_run(**kwargs)

    def setUp(self):
        pass
    def run(self):
        pass
    def tearDown(self):
        pass