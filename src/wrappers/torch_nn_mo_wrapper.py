from .torch_nn_wrapper import TorchNNWrapper
import torch
import tqdm
from callbacks import EarlyStopping, ModelCheckpoint
from requirements import RequirementSet
import os 

class EarlyStoppingException(Exception):
    pass

class TorchNNMOWrapper(TorchNNWrapper):
    """
    A wrapper class for a multi-objective neural network model using PyTorch.
    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
            - training_group_name (str): The name of the training group.
            - requirement_set (RequirementSet): The set of requirements for evaluation.
            - surrogate_functions (list[(callable, float)]): List of surrogate functions for evaluation.
    Attributes:
        training_group_name (str): The name of the training group.
        requirement_set (RequirementSet): The set of requirements for evaluation.
        surrogate_functions (list[(callable, float)]): List of surrogate functions for evaluation.
    Methods:
        _training_step(batch, batch_idx):
            Performs a single training step.
        _validation_step(batch, batch_idx):
            Performs a single validation step.
        _compute_kwargs(batch, outputs):
            Computes the keyword arguments required for evaluation.
        _evaluate_requirements(data_loader):
            Evaluates the requirements on the given data loader.
        _update_metrics(**kwargs):
            Updates the metrics for training and validation.
        fit(**kwargs):
            Trains the model for a specified number of epochs.
        score(data_loader, metrics, prefix=''):
            Scores the model on the given data loader using specified metrics.
    """
    def __init__(self,*args, **kwargs):
        super(TorchNNMOWrapper,self).__init__(*args, **kwargs)
        self.training_group_name:str = kwargs.get('training_group_name')
        self.requirement_set:RequirementSet = kwargs.get('requirement_set')
        self.surrogate_functions:list[(callable,float)] = kwargs.get('surrogate_functions')
        
        assert self.requirement_set is not None, f'{self.requirement_set} has to be provided'
        assert self.surrogate_functions is not None, f'{self.surrogate_functions} has to be provided'
        assert self.training_group_name is not None, f'{self.training_group_name} has to be provided'
    
    def _training_step(self,batch,batch_idx):
        self.model.train()
        inputs = batch['data'] 
        targets = batch['labels']
        group_ids = batch['groups']
        group_ids_list = batch['groups_ids_list']

        positive_mask=batch['positive_mask'].to(self.device)
        inputs = inputs.float().to(self.device)
        targets = targets.long().to(self.device)
        
        group_ids = {group_name:group_ids[group_name].to(self.device) for group_name in group_ids.keys()}
        group_ids_list = {group_name:group_ids_list[group_name].to(self.device) for group_name in group_ids_list.keys()}
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        
       
        loss = self.surrogate_functions.evaluate(logits=outputs,
                                                 labels=targets,
                                                 group_ids=group_ids,
                                                 positive_mask=positive_mask,
                                                 group_ids_list=group_ids_list,
                                                 group_masks=group_ids)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        #for param in self.model.parameters():
         #   print('Gradients: ',param.grad)
        self.optimizer.step()
        return loss.item()
    
    def _validation_step(self,batch,batch_idx):
        self.model.eval()
        with torch.no_grad():
            inputs = batch['data'] 
            targets = batch['labels']
            group_ids = batch['groups']
            group_ids_list = batch['groups_ids_list']
            positive_mask = batch['positive_mask'].to(self.device)

            
            inputs = inputs.float().to(self.device)
            targets = targets.long().to(self.device)
            group_ids = {group_name:group_ids[group_name].to(self.device) for group_name in group_ids.keys()}
            outputs = self.model(inputs)
            
        
            group_ids = {group_name:group_ids[group_name].to(self.device) for group_name in group_ids.keys()}
            group_ids_list = {group_name:group_ids_list[group_name].to(self.device) for group_name in group_ids_list.keys()}
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
    
            loss = self.surrogate_functions.evaluate(logits=outputs,
                                                    labels=targets,
                                                    group_ids=group_ids,
                                                    positive_mask=positive_mask,
                                                    group_ids_list=group_ids_list,
                                                    group_masks=group_ids)
            
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
           
            return loss.item(), outputs, targets, predictions
    
    def _compute_kwargs(self,batch,outputs):
        group_ids = batch['groups']
        group_ids_list = batch['groups_ids_list']
        positive_mask=batch['positive_mask'].to(self.device)
        group_ids = {group_name:group_ids[group_name].to(self.device) for group_name in group_ids.keys()}
        group_ids_list = {group_name:group_ids_list[group_name].to(self.device) for group_name in group_ids_list.keys()}
        labels = batch['labels'].to(self.device)
        kwargs = {
            'group_ids':group_ids,
            'positive_mask':positive_mask,
            'group_ids_list':group_ids_list,
            'logits':outputs,
            'group_masks':group_ids,
            'labels':labels
        }
        return kwargs
    
    def _evaluate_requirements(self,data_loader):
        with torch.no_grad():
            outputs = []
            targets = []
            groups = []
            predictions = []
            for batch_idx, batch in enumerate(data_loader):
                self.model.eval()
                loss, output, target,prediction = self._validation_step(batch, batch_idx)
                outputs.append(output)
                targets.append(target)
                predictions.append(prediction)
                groups.append(batch['groups'])
            
            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0).detach().cpu()
            predictions = torch.cat(predictions, dim=0).detach().cpu()
            groups_dict = {group_name:torch.cat([batch[group_name] for batch in groups],dim=0).detach().cpu() for group_name in groups[0].keys()}
            requirements,_,_ = self.requirement_set.evaluate(y_pred=predictions, 
                                                        y_true=targets, 
                                                        group_ids=groups_dict)
        return requirements,loss,outputs,targets,predictions,groups_dict
    
    def _update_metrics(self,**kwargs):
        self.model.eval()

        val_loader = self.data_module.val_loader()
        train_loader_eval = self.data_module.train_loader_eval()

        val_requirements,val_loss,val_outputs,val_targets,val_predictions,val_groups_dict = self._evaluate_requirements(val_loader)
        train_requirements,train_loss,train_outputs,train_targets,train_predictions,train_groups_dict = self._evaluate_requirements(train_loader_eval)
                
        metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_requirements': val_requirements,
                'train_requirements': train_requirements
                }
        
        metrics.update(self._compute_metrics(self.metrics,
                                                    val_predictions,
                                                    val_targets,
                                                    val_groups_dict,
                                                    prefix='val',
                                                    logits=val_outputs))
        metrics.update(self._compute_metrics(self.metrics,
                                                    train_predictions,
                                                    train_targets,
                                                    train_groups_dict,
                                                    prefix='train',
                                                    logits=train_outputs
                                                    ))
        return metrics

    
    def fit(self,**kwargs):
        
        num_epochs = kwargs.get('num_epochs',-1)
        disable_log = kwargs.get('disable_log',False)
        evaluate_best_model = kwargs.get('evaluate_best_model',True)
        n_rounds = self.num_epochs if num_epochs == -1 else num_epochs
        self.model.to(self.device)
        metrics = self._update_metrics()
        if not disable_log:
            self.logger.log(metrics)
        try:
            for epoch in tqdm.tqdm(range(n_rounds)):
                train_loader = self.data_module.train_loader()

                for batch_idx, batch in enumerate(tqdm.tqdm(train_loader)):
                    self._training_step(batch, batch_idx)
                
                metrics = self._update_metrics()
                
                for checkpoint in self.checkpoints:
                    if isinstance(checkpoint, EarlyStopping):
                        stop,counter = checkpoint(metrics=metrics)
                        metrics['early_stopping'] = counter
                        if stop:
                            if not disable_log:
                                self.logger.log(metrics)  
                            raise EarlyStoppingException 
                       
                    elif isinstance(checkpoint, ModelCheckpoint):
                        model_checkpoint= checkpoint(
                                save_fn=self.save, 
                                metrics=metrics)
                        metrics['model_checkpoint'] = 1 if model_checkpoint else 0
                if not disable_log:
                    self.logger.log(metrics)  
        except EarlyStoppingException:
                    pass
        
        for checkpoint in self.checkpoints:
                if isinstance(checkpoint, ModelCheckpoint):
                   if os.path.exists(checkpoint.get_model_path()):
                    self.load(checkpoint.get_model_path())

        if evaluate_best_model:
            metrics:dict = self._update_metrics()
            final_metrics = {}
            for name,value in metrics.items():
                final_metrics[f'final_{name}'] = value
            self.logger.log(final_metrics)
        
        return self.model

    def score(self, data_loader,metrics,prefix=''):
        assert len(data_loader) == 1, "Data loader should have a single batch"
        assert isinstance(metrics, list), "Metrics should be a list"
        self.model.to(self.device)
        scores = {}
        for batch_idx, batch in enumerate(data_loader):
            loss, outputs, targets, predictions = self._validation_step(batch, batch_idx)
            
            requirements,_,_ = self.requirement_set.evaluate(y_pred=predictions, 
                                                         y_true=targets, 
                                                         group_ids=batch['groups'])
            scores = self._compute_metrics(metrics,
                                           predictions,
                                           targets,
                                           batch['groups'],
                                           prefix=prefix,
                                           logits=outputs)
            
           
            if prefix != '':
                scores[f'{prefix}_requirements'] = requirements
                scores[f'{prefix}_loss'] = loss
            else: 
                scores['requirements'] = requirements
                scores['loss'] = loss
        return scores    

    
        