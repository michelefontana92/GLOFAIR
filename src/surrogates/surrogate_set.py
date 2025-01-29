from .base_surrogate import BaseSurrogate
from .demographic_parity import WassersteinDemographicParitySurrogate
import torch 
from torch.nn.functional import relu

class SurrogateFunctionSet:
    def __init__(self, surrogates: list[BaseSurrogate],**kwargs) -> None:
        self.surrogates: list[BaseSurrogate] = surrogates
        self.surrogate_dict: dict = {surrogate.name: surrogate for surrogate in surrogates}
        self.total_weight: float = sum([surrogate.weight for surrogate in surrogates])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.p = kwargs.get('p',2)
        self.weights = torch.stack([torch.tensor(surrogate.weight,dtype=torch.float) for surrogate in self.surrogates])
        print('self.weights:',self.weights)
    def evaluate(self,**kwargs):
        
        weights = self.weights.to(self.device)
        
        results_tensor = torch.stack([surrogate(**kwargs).to(self.device)
                        for surrogate in self.surrogates])
        #result = torch.sum(results_tensor)
        #print('results_tensor:',results_tensor)
        results_tensor = torch.pow(torch.abs(results_tensor),self.p)
        result = torch.pow(torch.dot(results_tensor,weights),1/self.p)
        
        """
        results_tensor = torch.stack([relu(surrogate(
                                            logits=logits, 
                                            labels=labels, 
                                            group_ids=group_ids_dict
                                            )-0.15) if (isinstance(surrogate,WassersteinDemographicParitySurrogate)) or (isinstance(surrogate,BaseSurrogate)) else surrogate(
                                            logits=logits, 
                                            labels=labels, 
                                            group_ids=group_ids_dict
                                            )
                        for surrogate in self.surrogates])
        
        result = torch.dot(results_tensor,weights)
        """
        return result
    
    def __str__(self) -> str:
        return '\n - \t '.join([str(surrogate) for surrogate in self.surrogates])