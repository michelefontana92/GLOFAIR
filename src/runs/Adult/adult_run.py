from ..base_run import BaseRun

class AdultRun(BaseRun):
    
    def __init__(self,**kwargs):
        super(AdultRun, self).__init__(**kwargs)
        self.input_size = 103
        self.hidden1 = 300
        self.hidden2 = 100
        self.dropout = 0.2
        
        
        self.dataset = 'adult'
        
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                               [
                                                ('Race',
                                                    {'race':['White','Black']}
                                                ),
                                                ('Gender',{'gender':['Male','Female']}),
                                                ('GenderRace',{
                                                    'race':['White','Black'],
                                                    'gender':['Male','Female']
                                                })
                                                ])
        
        self.init_run(**kwargs)
        
                        
    def setUp(self):
        pass
    def run(self):
        pass
    def tearDown(self):
        pass