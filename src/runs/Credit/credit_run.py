from ..base_run import BaseRun
from architectures import ArchitectureFactory
import shutil 

class CreditRun(BaseRun):
    
    def __init__(self,**kwargs):
        super(CreditRun, self).__init__(**kwargs)
        self.input_size = 92
        self.hidden1 = 300
        self.hidden2 = 100
        self.dropout = 0.2
        
        self.dataset = 'credit'
       
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                               [
                                                ('Age',
                                                    {'AGE':['20-40','40+']}
                                                ),
                                                ('Gender',{'SEX':['Male','Female']}),
                                                ('GenderAge',{
                                                    'AGE':['20-40','40+'],
                                                    'SEX':['Male','Female']
                                                })
                                                ])
        
        
        self.init_run(**kwargs)
    def setUp(self):
        pass
    
    def run(self):
        self.experiment.setup()
        self.experiment.run()

    def tearDown(self) -> None:
        shutil.rmtree(f'checkpoints/{self.project_name}')