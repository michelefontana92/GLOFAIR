#Define the torch_nn_wrapper.py file, which contains the TorchNNWrapper class.
# This class is a wrapper for a PyTorch neural network model. It inherits from the BaseWrapper class, which is defined in the base_wrapper.py file.
# The TorchNNWrapper class implements the fit, predict, predict_proba, score, save, load, get_params, set_params, get_feature_names, get_feature_types, get_feature_count, get_target_names, get_target_count, get_classes, and get_class_count methods.
from wrappers.base_wrapper import BaseWrapper
import torch 
import tqdm
import numpy as np
from loggers import BaseLogger
from callbacks import EarlyStopping, ModelCheckpoint
from dataloaders import DataModule
from loggers import WandbLogger,BaseLogger,FileLogger
from dataloaders import DataModule
from metrics import BaseMetric,Performance,GroupFairnessMetric
import tqdm
from sklearn.metrics import f1_score, classification_report,confusion_matrix
from torchmetrics import ConfusionMatrix,F1Score
import functools
from metrics import MetricsFactory
class EarlyStoppingException(Exception):
    pass

class TorchNNWrapper(BaseWrapper):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.model = kwargs.get('model')
        self.optimizer  = kwargs.get('optimizer')
        self.loss_fn = kwargs.get('loss')
        self.data_module:DataModule = kwargs.get('data_module')
        self.logger = kwargs.get('logger')
        self.num_epochs = kwargs.get('num_epochs')
        self.checkpoints = kwargs.get('checkpoints', [])
        self.metrics = kwargs.get('metrics', [])
        self.verbose = kwargs.get('verbose', False)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Assicurati che il modello sia inizializzato correttamente
       
        
        # Verifica che il modello abbia parametri per l'ottimizzatore
        assert len(list(self.model.parameters())) > 0, "Model has no parameters to optimize"

        self.loss = self.loss_fn()
        
        
        assert self.optimizer is not None, "Optimizer is required"
        assert self.model is not None, "Model is required"
        assert self.loss_fn is not None, "Loss function is required"
        assert self.logger is not None, "Logger is required"
        assert self.data_module is not None, "DataModule is required"
   
        assert isinstance(self.optimizer, torch.optim.Optimizer), "Optimizer should be an instance of torch.optim.Optimizer"
        assert issubclass(self.model.__class__, torch.nn.Module), "Model should be an instance of torch.nn.Module"
        assert issubclass(self.logger.__class__, BaseLogger), "Logger should be an instance of BaseLogger"
        assert issubclass(self.data_module.__class__, DataModule), "DataModule should be an instance of DataModule"
        assert isinstance(self.checkpoints, list), "Checkpoints should be a list of callbacks"
        for checkpoint in self.checkpoints:
            assert issubclass(checkpoint.__class__, EarlyStopping) or issubclass(checkpoint.__class__, ModelCheckpoint), "Checkpoints should be instances of EarlyStopping or ModelCheckpoint"
        
        assert isinstance(self.metrics, list), "Metrics should be a list"
        for metric in self.metrics:
            assert issubclass(metric.__class__, BaseMetric), "Metrics should be instances of BaseMetric"
        assert isinstance(self.num_epochs, int), "num_epochs should be an integer"
        assert isinstance(self.verbose, bool), "verbose should be a boolean"
        assert isinstance(self.device, torch.device), "device should be an instance of torch.device"
    
    def serialize(self):
        """
        Serializza tutti gli attributi essenziali della classe in un dizionario.
        """
         # Gestione della funzione di perdita
        if isinstance(self.loss_fn, functools.partial):
            loss_fn_name = self.loss_fn.func.__name__  # Nome della funzione originale
            loss_fn_args = self.loss_fn.args  # Argomenti posizionali
            loss_fn_kwargs = self.loss_fn.keywords  # Argomenti con parola chiave
        else:
            loss_fn_name = self.loss_fn.__name__ if self.loss_fn else None
            loss_fn_args = None
            loss_fn_kwargs = None

        
        checkpoints_serialized = [
        {'type': checkpoint.__class__.__name__, 'state': checkpoint.serialize()}
        for checkpoint in self.checkpoints
        ]
        return {
            'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn': {
                'name': loss_fn_name,
                'args': loss_fn_args,
                'kwargs': loss_fn_kwargs
            },
            'data_module': self.data_module.serialize(),
            'logger': self.logger.serialize(),
            'num_epochs': self.num_epochs,
            'checkpoints': checkpoints_serialized,
            'metrics': [MetricsFactory.serialize(metric) for metric in self.metrics],
            'verbose': self.verbose,
            'device': str(self.device),
        }


    @staticmethod
    def deserialize(data):
        """
        Ricostruisce un'istanza della classe `TorchNNWrapper` dai dati serializzati.
        """
        # Ricostruisci la funzione di perdita
        loss_fn_data = data['loss_fn']
        if loss_fn_data['name']:
            loss_fn = getattr(torch.nn, loss_fn_data['name'], None)
            if loss_fn_data['args'] or loss_fn_data['kwargs']:
                loss_fn = functools.partial(loss_fn, *loss_fn_data['args'], **loss_fn_data['kwargs'])
        
        # Ricostruisci l'istanza
        instance = TorchNNWrapper(
            model=torch.nn.Module(),
            optimizer=None,  # L'ottimizzatore sarà creato successivamente
            loss=loss_fn,
            data_module=DataModule.deserialize(data['data_module']),
            logger=WandbLogger.deserialize(data['logger']),
            num_epochs=data['num_epochs'],
            checkpoints=[],
            metrics=[BaseMetric.deserialize(metric) for metric in data['metrics']],
            verbose=data['verbose']
        )
        
        # Carica lo stato del modello e dell'ottimizzatore
        instance.model.load_state_dict(data['model_state_dict'])
        
        # Ricrea l'ottimizzatore con lo stato salvato
        optimizer_class = getattr(torch.optim, data['optimizer_state_dict']['class_name'])  # Recupera la classe dell'ottimizzatore
        instance.optimizer = optimizer_class(instance.model.parameters(), **data['optimizer_state_dict']['params'])
        instance.optimizer.load_state_dict(data['optimizer_state_dict']['state_dict'])

        instance.device = torch.device(data['device'])
        instance.model.to(instance.device)

         # Deserializza i checkpoint
        for checkpoint_data in data['checkpoints']:
            checkpoint_type = checkpoint_data['type']
            checkpoint_state = checkpoint_data['state']
            
            # Instanzia il checkpoint in base al tipo salvato
            if checkpoint_type == 'EarlyStopping':
                checkpoint = EarlyStopping.deserialize(checkpoint_state)
            elif checkpoint_type == 'ModelCheckpoint':
                checkpoint = ModelCheckpoint.deserialize(checkpoint_state)
            else:
                raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")
            
            instance.checkpoints.append(checkpoint)

        return instance

    
    def _training_step(self,batch,batch_idx):
        self.model.train()
        inputs = batch['data'] 
        targets = batch['labels']
       
        inputs = inputs.float().to(self.device)
        targets = targets.long().to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _validation_step(self,batch,batch_idx):
        self.model.eval()
        with torch.no_grad():
            inputs = batch['data']
            targets = batch['labels']
            inputs = inputs.float().to(self.device)
            targets = targets.long().to(self.device)
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
           
            return loss.item(), outputs, targets, predictions
    
    def _predict_step(self,batch,batch_idx):
        self.model.eval()
        with torch.no_grad():
            inputs = batch['data']
            inputs = inputs.float().to(self.device)
            outputs = self.model(inputs)
            return outputs
        
    def _compute_metrics(self,metrics,y_pred,y_true,
                         group_ids,prefix='val',**kwargs):
        tmp_result = {}
        final_result = {}
        logits = kwargs.get('logits',None)
      
        for metric in metrics:
            metric.reset()
            if issubclass(metric.__class__,GroupFairnessMetric):
                            group_ids_detached = {group_name:group_ids[group_name].detach().cpu() for group_name in group_ids.keys()}
                            metric.calculate(y_pred.detach().cpu(),
                                            y_true.detach().cpu(),
                                            group_ids_detached)
                            """
                            tmp_result.update(metric.compute_wasserstein_distance(logits=logits.detach().cpu(),
                                                                 labels=y_true.detach().cpu(),
                                                                 group_ids=group_ids_detached,
                                                                 use_differentiable=False))
                            """
            elif isinstance(metric,Performance):
                metric.calculate(y_pred.detach().cpu(),
                                 y_true.detach().cpu())
            else:
                raise ValueError(f"{metric} is an invalid metric")
            tmp_result.update(metric.get())
            
      
        for key, value in tmp_result.items():
            if prefix == '':
                final_result[key] = value
            else:
                final_result[f'{prefix}_{key}'] = value
        return final_result 
    
    def compute_stat_per_group(self,data_loader,
                                group_id,metric,
                                ):
     
        metric.reset()
        self.model.to(self.device)
        for batch_idx, batch in enumerate(data_loader):
            _, _, targets, predictions = self._validation_step(batch, batch_idx)
        if issubclass(metric.__class__,GroupFairnessMetric):
            metric.calculate(predictions.detach().cpu(),
                                            targets.detach().cpu(),
                                            batch['groups'].detach().cpu())
        else:
            raise ValueError(f"{metric} is an invalid metric")
        return metric.get_stats_per_group(group_id)
        

    # Fit the model
    def fit(self,**kwargs):
        weight = self.data_module.get_class_weights().to(self.device)
        self.loss = self.loss_fn(weight=weight)
        num_epochs = kwargs.get('num_epochs',-1)
        disable_log = kwargs.get('disable_log',True)
        n_rounds = self.num_epochs if num_epochs == -1 else num_epochs
        self.model.to(self.device)
        
        
        # self.model_checkpoint([self.model])

        try:
            for epoch in tqdm.tqdm(range(n_rounds)):
                train_loss = 0
                val_loss = 0
                train_loader = self.data_module.train_loader()
                val_loader = self.data_module.val_loader()
                train_loader_eval = self.data_module.train_loader_eval()
                for batch_idx, batch in enumerate(tqdm.tqdm(train_loader)):
                    train_loss += self._training_step(batch, batch_idx)
                
                train_loss /= len(train_loader)
                
                train_outputs = []
                train_targets = []
                train_predictions=[]
                train_groups = []
                
                val_outputs = []
                val_targets = []
                val_predictions = []
                val_groups = []
                for batch_idx, batch in enumerate(val_loader):
                    loss, outputs, targets,predictions = self._validation_step(batch, batch_idx)
                    val_loss += loss
                    val_outputs.append(outputs)
                    val_targets.append(targets)
                    val_predictions.append(predictions)
                    val_groups.append(batch['groups'])
                val_loss /= len(val_loader)
                
                for batch_idx, batch in enumerate(train_loader_eval):
                    _, outputs, targets,predictions = self._validation_step(batch, batch_idx)
                    train_outputs.append(outputs)
                    train_targets.append(targets)
                    train_predictions.append(predictions)
                    train_groups.append(batch['groups'])
                
                val_loss /= len(val_loader)
                val_outputs = torch.cat(val_outputs, dim=0)
                val_targets = torch.cat(val_targets, dim=0).detach().cpu()
                val_predictions = torch.cat(val_predictions, dim=0).detach().cpu()
                val_groups_dict = {group_name:torch.cat([batch[group_name] for batch in val_groups],dim=0).detach().cpu() for group_name in val_groups[0].keys()}
                #torch.cat(val_groups, dim=0).detach().cpu()
                train_outputs = torch.cat(train_outputs, dim=0)
                train_targets = torch.cat(train_targets, dim=0).detach().cpu()
                train_predictions = torch.cat(train_predictions, dim=0).detach().cpu()
                train_groups_dict = {group_name:torch.cat([batch[group_name] for batch in train_groups],dim=0).detach().cpu() for group_name in train_groups[0].keys()}

                metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
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
                                                    logits=train_outputs))
                
                
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
                   self.load(checkpoint.get_model_path())

        return self.model
    
    # Predict the target variable for the input data. The input data is a torch.Dataset object.
    def predict(self, data_loader):
        all_predictions=[]
        self.model.to(self.device)
        with torch.no_grad():
            self.model.eval()
            for batch_idx, batch in enumerate(data_loader):
                outputs = self._predict_step(batch, batch_idx)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions)
            return torch.cat(all_predictions, dim=0).detach().cpu()       
        
    
        
    # Predict the probabilities of the target variable for the input data     
    def predict_proba(self, data_loader):
        self.model.to(self.device)
        with torch.no_grad():
            self.model.eval()
            all_probabilities = []
            for batch_idx, batch in enumerate(data_loader):
                outputs = self._predict_step(batch, batch_idx)
                all_probabilities.append(torch.softmax(outputs, dim=1))
            return torch.cat(all_probabilities, dim=0).detach().cpu()

    def score(self, data_loader,metrics,prefix=''):
        assert len(data_loader) == 1, "Data loader should have a single batch"
        assert isinstance(metrics, list), "Metrics should be a list"
        self.model.to(self.device)
        scores = {}
        for batch_idx, batch in enumerate(data_loader):
            _, outputs, targets, predictions = self._validation_step(batch, batch_idx)
            scores = self._compute_metrics(metrics,
                                           predictions,
                                           targets,
                                           batch['groups'],
                                           prefix=prefix,
                                           logits=outputs)
        return scores        
    
    # Save the model to the specified path. The path should include the file name and extension.
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    # Load the model from the specified path. The path should include the file name and extension.
    def load(self, path):
       self.model.load_state_dict(torch.load(path))
    
    # Get the parameters of the model
    def get_params(self):
        return self.model.state_dict()
    
    # Set the parameters of the model
    def set_params(self,model_src):
        assert isinstance(model_src,torch.nn.Module), "model_src should be an instance of torch.nn.Module"
        model_dst_dict = self.model.state_dict()
        model_src_dict = model_src.state_dict()
        for key in model_src_dict.keys():
            model_dst_dict[key] = model_src_dict[key]
        self.model.load_state_dict(model_dst_dict)

    def set_params_from_dict(self,model_src_dict):
        assert isinstance(model_src_dict,dict), "model_src_dict should be a dictionary"
        model_dst_dict = self.model.state_dict()
        for key in model_src_dict.keys():
            model_dst_dict[key] = model_src_dict[key]
        self.model.load_state_dict(model_dst_dict)   
    
    def reset(self,optimizer_fn,callbacks_fn):
        self.optimizer = optimizer_fn(
              self.model.parameters()
              )
        self.checkpoints = [callback_fn() for callback_fn in callbacks_fn]

    def model_checkpoint(self,models_list):
        best_model_idx = 0
        original_model_dict = self.model.state_dict().copy()
        for idx,model in enumerate(models_list):
            if isinstance(model,torch.nn.Module):
                self.model = model
            elif isinstance(model,dict):
                self.set_params_from_dict(model)
            else:
                raise ValueError(f"Invalid model type {model}")
            val_loader = self.data_module.val_loader()
            initial_metrics = self.score(val_loader,self.metrics,prefix='val')
            for callback in self.checkpoints:
                if isinstance(callback, ModelCheckpoint):
                    if callback(metrics=initial_metrics,save_fn=self.save):
                        best_model_idx = idx
            self.model.load_state_dict(original_model_dict)
        return best_model_idx

    def get_train_loader(self):
        return self.data_module.train_loader()
    def get_val_loader(self):
        return self.data_module.val_loader()
    def get_train_loader_eval(self):
        return self.data_module.train_loader_eval()
    def get_group_ids(self):
        return self.data_module.get_group_ids()