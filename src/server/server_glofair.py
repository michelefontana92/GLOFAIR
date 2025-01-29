from .server_base import BaseServer
import ray

from .server_factory import register_server
from callbacks.early_stopping import EarlyStopping
from callbacks.model_checkpoint import ModelCheckpoint
from loggers.wandb_logger import WandbLogger
from functools import partial
from .aggregators import AggregatorFactory
import os
import numpy as np
import torch
import copy


class EarlyStoppingException(Exception):
    pass

@register_server("server_glofair")
class ServerGlofair(BaseServer):
    """
    ServerGlofair class for federated learning.
    Attributes:
        config (dict): Configuration dictionary.
        clients_init_fn_list (list): List of client initialization functions.
        model (torch.nn.Module): The model to be trained.
        loss (torch.nn.Module): Loss function.
        metrics (list): List of metrics for evaluation.
        log_model (bool): Flag to log the model.
        id (str): Server ID.
        project (str): Project name.
        checkpoint_dir (str): Directory to save checkpoints.
        checkpoint_name (str): Name of the checkpoint file.
        verbose (bool): Verbosity mode.
        federated_rounds (int): Number of federated learning rounds.
        callbacks (list): List of callbacks for training.
        logger (WandbLogger): Logger for tracking experiments.
        aggregator (Aggregator): Aggregator for federated learning.
    Methods:
        _create_clients(clients_init_fn_list):
            Creates and initializes clients.
        _broadcast_fn(fn_name, **kwargs):
            Broadcasts a function to all clients and collects results.
        _evaluate_best_model():
            Evaluates the best model saved during training.
        _evaluate_global_model(best_model=False):
            Evaluates the global model.
        setup(**kwargs):
            Sets up the server and clients.
        step(**kwargs):
            Executes one federated learning round.
        execute(**kwargs):
            Executes the federated learning process.
        centralized_training(**kwargs):
            Performs centralized training.
        evaluate(**kwargs):
            Evaluates the global model.
        fine_tune(**kwargs):
            Fine-tunes the global model.
        shutdown(**kwargs):
            Shuts down the server and evaluates the best model.
    """
    def __init__(self,**kwargs):
        
        self.config = kwargs.get('config')
        self.clients_init_fn_list = kwargs.get('clients_init_fn_list')
        self.model = kwargs.get('model')
        self.loss = kwargs.get('loss')
        self.metrics = kwargs.get('metrics')

        self.log_model = kwargs.get('log_model', False)
        self.id = kwargs.get('id', 'server_glofair')
        self.project = kwargs.get('project', 'server_glofair')
        self.checkpoint_dir = kwargs.get('checkpoint_dir','checkpoints')
        self.checkpoint_name = kwargs.get('checkpoint_name','global_model.h5')
       
        self.verbose = kwargs.get('verbose', False)
        self.federated_rounds = self.config['num_rounds']
        
        
        self.callbacks = [
            EarlyStopping(patience=self.config['early_stopping_patience'],
                          monitor='global_val_requirements',
                          mode='min'
                          ),
            ModelCheckpoint(save_dir=self.checkpoint_dir,
                            save_name = self.checkpoint_name,
                            monitor='global_val_requirements',
                            mode='min')
                          ]
        
        self.logger = WandbLogger(
            project=self.project,
            config= self.config,
            id=self.id,
            checkpoint_dir= self.checkpoint_dir,
            checkpoint_path = self.checkpoint_name,
            data_module=self.data if self.log_model else None
        )

        self.aggregator = AggregatorFactory().create('FedAvgAggregator')
 
    def _create_clients(self,clients_init_fn_list):
        client_list = [client_init_fn() 
                for client_init_fn in clients_init_fn_list]
        print('Clients:',client_list)
        return client_list
    
    
    def _broadcast_fn(self,fn_name,**kwargs):
        assert isinstance(fn_name,str), "fn_name must be a string"
        handlers = []
        results = []
        for client in self.clients:
            assert hasattr(client,fn_name), f"Client does not have {fn_name} method"
            handlers.append(getattr(client,fn_name).remote(**kwargs))
        for handler in handlers:
            results.append(ray.get(handler))
        return results
    
    def _evaluate_best_model(self):
        global_model = torch.load(self.checkpoint_path)
        self.model.load_state_dict(global_model)
        global_scores = self._evaluate_global_model(best_model=True)
        final_scores ={}
        for key,v in global_scores.items():
            final_scores[f'final_{key}'] = v
        self.logger.log(final_scores)

    def _evaluate_global_model(self,best_model=False):
        if best_model:
            scores = self._broadcast_fn('evaluate_best_model',
                            global_model=self.model)
        else: 
            scores = self._broadcast_fn('evaluate',
                            global_model=self.model)    
        global_scores = {}
        for score in scores:
            for kind in score.keys():
                for metric in score[kind].keys():
                    name = f'global_{kind}_{metric}'
                    if name not in global_scores:
                        global_scores[name] = []
                    global_scores[name].append(score[kind][metric])
        for metric in global_scores:
            global_scores[metric] = np.mean(global_scores[metric])

        return global_scores

    
    def setup(self,**kwargs):
        self.checkpoint_path = os.path.join(self.checkpoint_dir,self.checkpoint_name)
        self.clients = self._create_clients(
            self.clients_init_fn_list)
        self._broadcast_fn('setup',
                           global_model_ckpt_path=self.checkpoint_path)
    
    
    def step(self,**kwargs):
        results = self._broadcast_fn('update',
                           global_model=self.model)
        global_model = copy.deepcopy(self.model)
        new_params = self.aggregator(model=global_model,
                        params=results)
        self.model.load_state_dict(new_params)
        global_scores = self._evaluate_global_model()
        global_scores['global_round'] = kwargs.get('round') + 1
        try:
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopping):
                    stop,counter = callback(metrics=global_scores)
                    global_scores['global_early_stopping'] = counter
                    if stop:
                        self.logger.log(global_scores)  
                        raise EarlyStoppingException  
                elif isinstance(callback,ModelCheckpoint):
                    callback(save_fn=partial(torch.save, self.model.state_dict()),
                            metrics = global_scores)
            self.logger.log(global_scores)
        
        except EarlyStoppingException:
            raise EarlyStoppingException 
                
    def execute(self,**kwargs):
        try:
            for i in range(self.federated_rounds):
                self.step(round=i)
        except EarlyStoppingException:
            pass
        self.fine_tune()

    def centralized_training(self,**kwargs):
        self._broadcast_fn('update',
                           global_model=self.model)
        self._evaluate_global_model()

    def evaluate(self,**kwargs):
        global_scores = self._evaluate_global_model()
        return global_scores
    
    def fine_tune(self,**kwargs):
        if os.path.exists(self.checkpoint_path):
            global_model = torch.load(self.checkpoint_path)
            self.model.load_state_dict(global_model)
        self._broadcast_fn('fine_tune',global_model=self.model)
        return
    
    def shutdown(self,**kwargs):
        if os.path.exists(self.checkpoint_path):
            self._evaluate_best_model()
        self.logger.close()
        self._broadcast_fn('shutdown')