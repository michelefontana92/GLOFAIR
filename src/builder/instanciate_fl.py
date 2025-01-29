from server import ServerFactory
from client import ClientFactory
from functools import partial
from requirements import RequirementSet
from surrogates import SurrogateFunctionSet
import ray 
import os 


class GLOFAIR_Builder:

    def _assign_resources(self):
        num_clients = self.num_clients
        self.num_cpus = num_clients + 1
        self.num_gpus = len(self.gpu_devices)
     
        self.num_gpus_per_client = self.num_gpus//num_clients if self.num_gpus > 0 else 0
        
    
    def __init__(self,**kwargs):
        self.num_clients = kwargs.get('num_clients')
        self.gpu_devices = kwargs.get('gpu_devices',[]) 
        self._assign_resources()
        self._server_config = kwargs.get('server_config')
        self._clients_config = kwargs.get('clients_config')
        self._algorithm_config = kwargs.get('algorithm_config')
        self._project = kwargs.get('project')
        
        assert self._server_config is not None, 'Server config is None'
        assert self._clients_config is not None, 'Clients config is None'
        assert self._algorithm_config is not None, 'Algorithm config is None'

        assert isinstance(self._server_config,dict), "server_config must be a dictionary"
        assert isinstance(self._clients_config,list), "clients_config must be a list"
        assert isinstance(self._algorithm_config,dict), "algorithm_config must be a dictionary"
        
        
        
        self.model = self._algorithm_config.get('model')
        self.loss = self._algorithm_config.get('loss')
        self.metrics = self._algorithm_config.get('metrics')
        self.optimizer_name = self._algorithm_config.get('optimizer_name','Adam')
        self.lr = self._algorithm_config.get('lr',1e-3)
        self.training_group_name = self._algorithm_config.get('training_group_name')
        self.num_local_epochs = self._algorithm_config.get('num_local_epochs',3)
        self.num_federated_epochs = self._algorithm_config.get('num_federated_epochs',2)
        self.fine_tune_epochs = self._algorithm_config.get('fine_tune_epochs',50)
        self.verbose = self._algorithm_config.get('verbose',False)
        self.surrogate_set:SurrogateFunctionSet = self._algorithm_config.get('surrogate_set')
        self.requirement_set:RequirementSet = self._algorithm_config.get('requirement_set')
        self.clients = []
        
        for i,client_config in enumerate(self._clients_config):
            c_id = client_config.get('id',f'glofair_client_{i+1}')
            c_data = client_config.get('data')
            c_ckpt_dir = client_config.get('checkpoint_dir')
            c_ckpt_name = client_config.get('checkpoint_name')
            c_patience = client_config.get('early_stopping_patience')
            c_monitor = client_config.get('monitor')
            c_mode = client_config.get('mode')
            assert c_data is not None, 'Data is None'
            assert c_ckpt_dir is not None, 'Checkpoint dir is None'
            assert c_ckpt_name is not None, 'Checkpoint name is None'
            assert c_patience is not None, 'Patience is None'
            assert c_monitor is not None, 'Monitor is None'
            assert c_mode is not None, 'Mode is None'
            
            c_config = {
                'early_stopping_patience': c_patience,
                'monitor': c_monitor,
                'mode': c_mode,
                'optimizer_name': self.optimizer_name,
                'lr': self.lr,
                'num_epochs':self.num_local_epochs
                
            }


            self.clients.append(partial(ClientFactory().create,
                'client_glofair',
                remote=True,
                num_gpus=self.num_gpus_per_client,
                config=c_config,
                id = c_id,
                project = self._project,
                model = self.model,
                data = c_data,
                loss = self.loss,
                metrics = self.metrics,
                log_model = client_config.get('log_model',False),
                checkpoint_dir = c_ckpt_dir,
                checkpoint_name = c_ckpt_name,
                training_group_name = self.training_group_name,
                surrogate_set = self.surrogate_set,
                requirement_set = self.requirement_set,
                fine_tune_epochs = self.fine_tune_epochs
                )
                )
            
        s_patience = self._server_config.get('early_stopping_patience')
        s_monitor = self._server_config.get('monitor')
        s_mode = self._server_config.get('mode')
        assert s_patience is not None, 'Patience is None'
        assert s_monitor is not None, 'Monitor is None'
        assert s_mode is not None, 'Mode is None'
        s_config = {
            'early_stopping_patience': s_patience,
            'monitor': s_monitor,
            'mode': s_mode,
            'num_rounds':self.num_federated_epochs
        }

        s_id = self._server_config.get('id','glofair_server')
        s_ckpt_dir = self._server_config.get('checkpoint_dir',f'checkpoints/{self._project}')
        s_ckpt_name = self._server_config.get('checkpoint_name','glofair_global_model.h5')
        
        self.server = ServerFactory().create('server_glofair',
        config=s_config,
        id = s_id,
        project = self._project,
        model = self.model,
        metrics = self.metrics,
        loss = self.loss,
        clients_init_fn_list = self.clients,
        log_model = self._server_config.get('log_model',False),
        checkpoint_dir = s_ckpt_dir,
        checkpoint_name = s_ckpt_name,
        verbose = self.verbose,
        
        )
    
    def run(self):
        print('Number of CPUs:',self.num_cpus)
        print('Number of GPUs:',self.num_gpus)
        print('Number of GPUs per client:',self.num_gpus_per_client)
        ray.init(num_cpus=self.num_cpus,num_gpus=self.num_gpus)
        self.server.setup()
        self.server.execute()
        self.server.shutdown()
        ray.shutdown()
     