from .experiment import Experiment
from dataloaders import DataModule
from algorithms import AlgorithmFactory
from metrics import MetricsFactory
import torch
from functools import partial 

class GlofairExperiment(Experiment):
    def __init__(self,**kwargs):
        
        self.model = kwargs.get('model')

        self.dataset = kwargs.get('dataset','adult')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                               {'gender':['Male','Female']})
        
        self.data_root = kwargs.get('data_root','data/Adult')
        self.start_index = kwargs.get('start_index',1)
        self.num_clients = kwargs.get('num_clients',2)
        self.batch_size = kwargs.get('batch_size',128)
        self.num_workers = kwargs.get('num_workers',1)
        self.data_path_list = kwargs.get('data_path',[])
        self.training_group_name = kwargs.get('training_group_name')
        self.surrogate_set = kwargs.get('surrogate_set')
        self.requirement_set = kwargs.get('requirement_set')
        self.start_index = kwargs.get('start_index',51)
        self.num_federated_rounds = kwargs.get('num_federated_rounds',30)
        self.num_local_epochs = kwargs.get('num_local_epochs',3)
        self.fine_tune_epochs = kwargs.get('fine_tune_epochs',50)
        self.lr = kwargs.get('lr',1e-4)
        self.optimizer_name = kwargs.get('optimizer_name','Adam')
        
        self.project = kwargs.get('project','Glofair')

        self.server_config = {}
        self.clients_config = []
        self.algorithm_config = {}
        
        if len(self.data_path_list) == 0:
            for i in range(self.num_clients): 
                    data_path = 'node_'+str(self.start_index+i)
                    self.clients_config.append({
                        'id': f'glofair_client_{i+1}',
                        'data': DataModule(
                            dataset=self.dataset,
                            root=self.data_root,
                            train_set=f'{data_path}/{self.dataset}_train.csv',
                            val_set=f'{data_path}/{self.dataset}_val.csv',
                            test_set=f'{data_path}/{self.dataset}_val.csv',
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            sensitive_attributes=self.sensitive_attributes
                        )
                    })
        else:
            for i,data_path in self.data_path_list:
                self.clients_config.append({
                    'id': f'glofair_client_{i+1}',
                    'data': DataModule(
                        dataset=self.dataset,
                        root=self.data_root,
                        train_set=f'{data_path}/{self.dataset}_train.csv',
                        val_set=f'{data_path}/{self.dataset}_val.csv',
                        test_set=f'{data_path}/{self.dataset}_val.csv',
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        sensitive_attributes=self.sensitive_attributes
                    )
                }) 

        
        self.group_ids = self.compute_groups_ids(self.clients_config)
        self.server_config = {
            'early_stopping_patience':5,
            'monitor':'global_val_requirements',
            'mode':'min'
        }
        
        self.metrics = [MetricsFactory().create_metric('performance')]
        for group_name in self.group_ids.keys():
            self.metrics += [
                        MetricsFactory().create_metric('demographic_parity',
                                                       group_ids=self.group_ids,
                                                       group_name = group_name),
                        MetricsFactory().create_metric('equal_opportunity',
                                                       group_ids=self.group_ids,
                                                       group_name = group_name),
                        MetricsFactory().create_metric('equalized_odds',
                                                       group_ids=self.group_ids,
                                                       group_name = group_name)
            ]

        self.algorithm_config={
            'model': self.model,
            'loss': partial(torch.nn.CrossEntropyLoss),
            'metrics': self.metrics,
            'optimizer_name': self.optimizer_name,
            'lr': self.lr,
            'num_local_epochs': self.num_local_epochs,
            'fine_tune_epochs': self.fine_tune_epochs,
            'num_federated_epochs': self.num_federated_rounds,
            'verbose': False,
            'training_group_name':self.training_group_name,
            'surrogate_set':self.surrogate_set,
            'requirement_set':self.requirement_set
        }

       
    def setup(self):
        self.algorithm = AlgorithmFactory().create('Glofair',
                                                  project=self.project,
                                                  server_config=self.server_config,
                                                  clients_config=self.clients_config,
                                                  algorithm_config=self.algorithm_config)

    def run(self):
        self.algorithm()
    
    def teardown(self, **kwargs):
        pass