from ..base_run import BaseRun
from architectures import ArchitectureFactory
import shutil 

class CreditRun(BaseRun):
    
    def __init__(self,**kwargs):
        super(CreditRun, self).__init__(**kwargs)
        self.hidden1 = 300
        self.hidden2 = 100
        self.dropout = 0.2
        self.model = ArchitectureFactory.create_architecture('mlp2hidden',model_params={
                                                'input': 92,
                                                'hidden1': self.hidden1,
                                                'hidden2':self.hidden2,
                                                'dropout': self.dropout,
                                                'output': 2})
        self.dataset = 'credit'
        self.data_root  = 'data/Credit'
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
        
        
        self.num_clients = 10
        self.lr=1e-4
        self.num_federated_rounds = 100
    
    def setUp(self):
        pass
    
    def run(self):
        self.experiment.setup()
        self.experiment.run()

    def tearDown(self) -> None:
        shutil.rmtree(f'checkpoints/{self.project_name}')