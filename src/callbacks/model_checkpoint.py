import os

class ModelCheckpoint:
    def __init__(self, save_dir, save_name, 
                 monitor='val_loss', mode='min',
                 check_fn = None):
        self.save_dir = save_dir
        self.save_name = save_name
        self.monitor = monitor
        self.mode = mode
        self.best = None
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        if check_fn is not None:
            assert callable(check_fn), "Check Function must be callable"
            assert len(check_fn.__code__.co_varnames) == 1, "Check Function must have only one argument" 
            self.check = check_fn

    def set_check_fn(self, check_fn:callable):
        assert callable(check_fn), "Check Function must be callable"
        assert len(check_fn.__code__.co_varnames) == 1, "Check Function must have only one argument" 
        self.check = check_fn

    def check(self,**kwargs):
        metrics = kwargs.get('metrics')
        if self.best is None or (self.mode == 'min' and metrics[self.monitor] < self.best) or (self.mode == 'max' and metrics[self.monitor] > self.best):    
            return True
        return False
    
    def __call__(self, **kwargs):
        save_fn = kwargs.get('save_fn')
        assert save_fn is not None, "Save Function is required for ModelCheckpoint"
        metrics = kwargs.get('metrics')
        assert metrics is not None, "Metrics are required for ModelCheckpoint"
        assert isinstance(metrics, dict), "Metrics must be a dictionary"
        if self.check(metrics=metrics):
            self.best = metrics[self.monitor]         
            save_fn(os.path.join(self.save_dir, self.save_name))
            return True
        return False
    
    def get_model_path(self):
        return os.path.join(self.save_dir, self.save_name)
    
    def get_best(self):
        return self.best
    
    def get_best_metric(self):
        return {self.monitor:self.best}
    
    def reset(self):
        self.best = None

    def serialize(self):
        """
        Serializza l'oggetto ModelCheckpoint in un dizionario.
        """
        return {
            'save_dir': self.save_dir,
            'save_name': self.save_name,
            'monitor': self.monitor,
            'mode': self.mode,
            'best': self.best
        }

    @staticmethod
    def deserialize(data):
        """
        Ricostruisce un oggetto ModelCheckpoint dai dati serializzati.
        """
        instance = ModelCheckpoint(
            save_dir=data['save_dir'],
            save_name=data['save_name'],
            monitor=data['monitor'],
            mode=data['mode']
        )
        instance.best = data['best']
        return instance