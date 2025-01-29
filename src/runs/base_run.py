from abc import ABC, abstractmethod
from architectures import ArchitectureFactory
from metrics import MetricsFactory
from dataloaders import DataModule
import torch
from functools import partial
from surrogates import SurrogateFactory,SurrogateFunctionSet
from requirements import RequirementSet, UnconstrainedRequirement, ConstrainedRequirement
from builder import GLOFAIR_Builder
class BaseRun(ABC):
    def __init__(self,**kwargs):
        
        self.model = kwargs.get('model')
        self.dataset = kwargs.get('dataset')
        self.sensitive_attributes = kwargs.get('sensitive_attributes')
        self.project_name = kwargs.get('project_name')
        self.data_root = kwargs.get('data_root')
        self.gpu_devices = kwargs.get('gpu_devices')

    def compute_group_cardinality(self,group_name):
        for name,group_dict in self.sensitive_attributes:
            if name == group_name:
                total = 1
                for key in group_dict.keys():
                    total *= len(group_dict[key])
                return total 
        raise KeyError(f'Group {group_name} not found in sensitive attributes') 
    
    def init_run(self,**kwargs):
        self.experiment=kwargs.get('experiment')
        self.num_clients = kwargs.get('num_clients')
      
        self.metrics_list = kwargs.get('metrics_list')
        self.groups_list = kwargs.get('groups_list')
        self.threshold_list = kwargs.get('threshold_list')
        
        self.project_name =  kwargs.get('project_name')
        self.use_wandb = kwargs.get('use_wandb')
        
        self.data_root  = f'../data/{self.dataset.capitalize()}/{self.experiment}'
        print(self.data_root)
        self.model = ArchitectureFactory.create_architecture('mlp2hidden',model_params={
                                                'input': self.input_size,
                                                'hidden1': self.hidden1,
                                                'hidden2': self.hidden2,
                                                'dropout': self.dropout,
                                                'output': 2})
       
        
       
        self.metrics = [MetricsFactory().create_metric('performance')]
        self.approximations = [SurrogateFactory().create(name='performance',
                                                         surrogate_name='approx_performance', 
                                                         surrogate_weight=1)]
        self.requirements = [
            UnconstrainedRequirement(name='unconstrained_performance_requirement',
                             metric = MetricsFactory.create_metric(
                                    metric_name='performance'),
                             weight=1,
                             mode='max',
                             bound=1.0,
                             performance_metric='f1'
                             )]
        
        self.callbacks = []
        
        
        for metric, group, threshold in zip(self.metrics_list, self.groups_list, self.threshold_list):
           
            self.metric = metric
            self.training_group_name = group
            self.num_groups = self.compute_group_cardinality(self.training_group_name)
            self.group_ids = {self.training_group_name: list(range(self.num_groups))}
            
            # Aggiunta della metrica
            current_metric = MetricsFactory().create_metric(metric, group_ids=self.group_ids, group_name=self.training_group_name, 
                                                            )

            approximation = SurrogateFactory.create(name=f'{self.metric}', 
                                                                surrogate_name=f'approx_{self.metric}_{group}', 
                                                                surrogate_weight=2, 
                                                                reduction='mean', 
                                                                group_name=group, 
                                                                unique_group_ids={group: list(range(self.num_groups))}
                                                                )
            
            requirement = ConstrainedRequirement(name=f'{metric}_requirement',
                                                 metric = current_metric,
                                                 weight=1, 
                                                 operator='<=',
                                                 threshold=threshold)       
            self.approximations.append(approximation)
            self.metrics.append(current_metric)
            self.requirements.append(requirement)
            
        self.surrogate_function_set = SurrogateFunctionSet(self.approximations)
        self.requirement_set = RequirementSet(self.requirements)
    
    def build_server_config(self,**kwargs):
        self.server_config = {
            'early_stopping_patience':5,
            'monitor':'global_val_requirements',
            'mode':'min'
        }

    def build_client_config(self,**kwargs):
        self.clients_config = []
        for i in range(1,self.num_clients+1): 
            data_path = 'node_'+str(i)
            self.clients_config.append({
                'id': f'glofair_client_{i}',
                'data': DataModule(
                            dataset=self.dataset,
                            root=self.data_root,
                            train_set=f'{data_path}/{self.dataset}_train.csv',
                            val_set=f'{data_path}/{self.dataset}_val.csv',
                            test_set=f'{data_path}/{self.dataset}_val.csv',
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            sensitive_attributes=self.sensitive_attributes
                        ),
                'checkpoint_dir': f'checkpoints/{self.project_name}',
                'checkpoint_name': f'glofair_local_model_{i}.h5',
                'early_stopping_patience': self.local_early_stopping_patience,
                'monitor': 'val_requirements',
                'mode': 'min',
                'log_model': False

                    })
    
    def build_algorithm_config(self,**kwargs):
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
            'surrogate_set':self.surrogate_function_set,
            'requirement_set':self.requirement_set

        }
    
    def init_fl(self,**kwargs):
        self.build_server_config(**kwargs)
        self.build_client_config(**kwargs)
        self.build_algorithm_config(**kwargs)
        print('Server config:',self.server_config)
        print('Algorithm config:',self.algorithm_config)
        for c in self.clients_config:
            print(c)
        
        builder = GLOFAIR_Builder(server_config=self.server_config,
                                       clients_config=self.clients_config,
                                      algorithm_config=self.algorithm_config,
                                      project=self.project_name,
                                      gpu_devices=self.gpu_devices,
                                      num_clients=self.num_clients,)
        
        print('FL initialized')
        print()
        return builder
        
    
    
    @abstractmethod
    def setUp(self):
        pass 
    
    @abstractmethod
    def tearDown(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def __call__(self):
        self.setUp()
        self.run()
        self.tearDown()
