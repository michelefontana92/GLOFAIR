from .adult_run import AdultRun
from ..run_factory import register_run

@register_run('adult_glofair')
class AdultGlofair(AdultRun):
    def __init__(self,**kwargs) -> None:
        super(AdultGlofair, self).__init__(**kwargs)
        self.lr=1e-4
        self.num_federated_rounds = 2
        self.num_local_epochs = 3
        self.fine_tune_epochs = 1
        self.batch_size = 128
        self.num_workers = 1
        self.optimizer_name='Adam'
        self.local_early_stopping_patience=3
        self.builder = self.init_fl(**kwargs)
    
    def run(self):
        self.builder.run()
        
