from .client_base import BaseClient
import ray
from loggers.wandb_logger import WandbLogger
from callbacks import EarlyStopping, ModelCheckpoint
from wrappers import TorchNNMOWrapper
from functools import partial
from .client_factory import register_client
import torch
import os
from surrogates import SurrogateFunctionSet
from requirements import RequirementSet

@register_client("client_glofair")
@ray.remote(num_cpus=1,num_gpus=0.1)
class ClientGlofair(BaseClient):
    
    def __init__(self, config,data, model, 
                 loss, metrics,**kwargs):
        self.config = config
        self.model = model 
        self.data = data
        self.log_model = kwargs.get('log_model', False)
        self.id = kwargs.get('id', 'client_glofair')
        self.project = kwargs.get('project', 'client_glofair')
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 
                                         f'checkpoints/{self.project}')
        self.checkpoint_name = kwargs.get('checkpoint_name',
                                           'model.h5')
        self.optimizer_name = kwargs.get('optimizer_name','Adam')
        self.verbose = kwargs.get('verbose', False)
        self.fine_tune_epochs = kwargs.get('fine_tune_epochs', 50)
        self.loss = loss
        self.metrics = metrics
        self.optimizer_fn = partial(getattr(torch.optim,
                                            self.optimizer_name),
                                            lr=self.config['lr'])
        self.checkpoint_path = os.path.join(self.checkpoint_dir,self.checkpoint_name)
        self.callbacks_fn = [
            partial(EarlyStopping,
                    patience=self.config['early_stopping_patience'],
                    monitor='val_requirements',
                    mode='min'
                    ),
            partial(ModelCheckpoint,
                    save_dir=self.checkpoint_dir,
                    save_name = self.checkpoint_name,
                    monitor='val_requirements',
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

        self.surrogate_set:SurrogateFunctionSet = kwargs.get('surrogate_set')
        self.requirement_set:RequirementSet = kwargs.get('requirement_set')
        assert self.surrogate_set is not None, "surrogate_set must be provided"
        assert self.requirement_set is not None, "requirement_set must be provided"
        self.training_group_name:str = kwargs.get('training_group_name')
        assert self.training_group_name is not None, "training_group_name must be provided"
    
    def setup(self,**kwargs):
        self.global_model_ckpt_path= kwargs.get('global_model_ckpt_path')

        self.wrapper = TorchNNMOWrapper(
            model=self.model,
            optimizer=self.optimizer_fn(self.model.parameters()),
            loss=self.loss,
            data_module=self.data,
            logger=self.logger,
            checkpoints=[checkpoint_fn() for checkpoint_fn in self.callbacks_fn],
            metrics=self.metrics,
            num_epochs=self.config['num_epochs'],
            verbose=self.verbose,
            training_group_name=self.training_group_name,
            requirement_set=self.requirement_set,
            surrogate_functions=self.surrogate_set
            
           )

    def update(self,**kwargs):
        global_model = kwargs.get('global_model')
        assert isinstance(global_model,torch.nn.Module), "global_model must be a torch.nn.Module"
        self.wrapper.reset(self.optimizer_fn,self.callbacks_fn)
        self.wrapper.set_params(global_model)
        self.wrapper.fit()
        
        result = {
            'weight': len(self.wrapper.data_module.datasets['train']),
            'params': self.wrapper.get_params()
        }
       

        return result
        
    def evaluate(self,**kwargs):
        local_scores = self._evaluate_local_model()
        dict2send, global_scores = self._evaluate_global_model(**kwargs)
        global_scores.update(local_scores)
        self.wrapper.logger.log(global_scores)
        return dict2send
    
    def evaluate_best_model(self,**kwargs):
        local_scores = self._evaluate_local_model()
        dict2send, global_scores = self._evaluate_global_model(**kwargs)
        global_scores.update(local_scores)
        final_scores = {}
        for key,v in global_scores.items():
            final_scores[f'final_{key}'] = v
        self.wrapper.logger.log(final_scores)
        return dict2send
    
    def _evaluate_global_model(self,**kwargs):
        global_model = kwargs.get('global_model')
        assert isinstance(global_model,torch.nn.Module), "global_model must be a torch.nn.Module"
        self.wrapper.set_params(global_model)
        train_scores=self.wrapper.score(
            self.wrapper.data_module.train_loader_eval(),
            self.metrics)
        val_scores=self.wrapper.score(
            self.wrapper.data_module.val_loader(),
            self.metrics)
        local_metrics = {}
        for metric in train_scores.keys():
            local_metrics[f'global_train_{metric}'] = train_scores[metric]
            local_metrics[f'global_val_{metric}'] = val_scores[metric]
        return {'train':train_scores,'val':val_scores},local_metrics
    
    def _evaluate_local_model(self,**kwargs):
        local_model = torch.load(self.checkpoint_path)
        self.wrapper.set_params_from_dict(local_model)
        train_scores=self.wrapper.score(
            self.wrapper.data_module.train_loader_eval(),
            self.metrics)
        val_scores=self.wrapper.score(
            self.wrapper.data_module.val_loader(),
            self.metrics)
        local_metrics = {}
        for metric in train_scores.keys():
            local_metrics[f'local_train_{metric}'] = train_scores[metric]
            local_metrics[f'local_val_{metric}'] = val_scores[metric]
        return local_metrics

    def fine_tune(self,**kwargs):
        global_model = kwargs.get('global_model')        
        assert isinstance(global_model,torch.nn.Module), "global_model must be a torch.nn.Module"
        if os.path.exists(self.checkpoint_path):
            local_model = torch.load(self.checkpoint_path)
            models = [global_model,local_model]
        else: 
            models = [global_model]
        self.wrapper.reset(self.optimizer_fn,self.callbacks_fn)
        best_idx = self.wrapper.model_checkpoint(models)
        best_model = models[best_idx]
        if isinstance(best_model,dict):
            self.wrapper.set_params_from_dict(best_model)
        else:
            self.wrapper.set_params(best_model)
        self.wrapper.fit(num_epochs=self.fine_tune_epochs)
        local_scores = self._evaluate_local_model()
        self.wrapper.logger.log(local_scores)

    def shutdown(self,**kwargs):
        self.wrapper.logger.close()

