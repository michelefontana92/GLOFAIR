from .credit_run import CreditRun
from ..run_factory import register_run

@register_run('credit_glofair')
class CreditGlofair(CreditRun):
    def __init__(self,**kwargs) -> None:
        super(CreditGlofair, self).__init__(**kwargs)
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
        
