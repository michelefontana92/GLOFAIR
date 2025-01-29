class EarlyStopping:
    def __init__(self, patience=5, delta=0.0,
                 monitor='val_loss', mode='min'):
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, **kwargs):
        metrics = kwargs.get('metrics')
        assert metrics is not None, "Metrics are required for EarlyStopping"
        assert isinstance(metrics, dict), "Logs must be a dictionary"
        score = metrics[self.monitor]
        if (self.best_score is not None) and ((self.mode == 'min' and score > self.best_score) or (self.mode == 'max' and score < self.best_score)):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop,self.counter

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def serialize(self):
        """
        Serializza lo stato dell'oggetto EarlyStopping in un dizionario.
        """
        return {
            'patience': self.patience,
            'delta': self.delta,
            'monitor': self.monitor,
            'mode': self.mode,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop
        }

    @staticmethod
    def deserialize(data):
        """
        Ricostruisce un oggetto EarlyStopping dallo stato serializzato.
        """
        instance = EarlyStopping(
            patience=data['patience'],
            delta=data['delta'],
            monitor=data['monitor'],
            mode=data['mode']
        )
        instance.counter = data['counter']
        instance.best_score = data['best_score']
        instance.early_stop = data['early_stop']
        return instance