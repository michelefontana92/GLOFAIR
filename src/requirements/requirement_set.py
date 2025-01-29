from .base_requirement import BaseRequirement
from .constrained_requirement import ConstrainedRequirement
class RequirementSet:

    def __init__(self, requirements: list[BaseRequirement],**kwargs) -> None:
        self.requirements: list[BaseRequirement] = requirements
        self.requirement_dict: dict = {requirement.name: requirement for requirement in requirements}
        self.total_weight: float = sum([requirement.weight for requirement in requirements])
        self.penalty = kwargs.get('penalty',100.0)
    
    def evaluate(self,y_pred, y_true, group_ids):    
       
        results_dict = {}
        hard_constraints_satified = True
        for requirement in self.requirements:
            eval_dict = requirement.evaluate(y_pred, y_true, group_ids)
            current_penalty = 1.0
            if isinstance(requirement,ConstrainedRequirement) and requirement.hard_constraint and not eval_dict['status']:
                current_penalty = self.penalty
                hard_constraints_satified = False
            results_dict.update({requirement.name: eval_dict[requirement.name]*current_penalty})
            
        total_distance = round(sum([result for result in results_dict.values()]) / self.total_weight,3)
        
        return total_distance,results_dict,hard_constraints_satified
    
    
    def __str__(self) -> str:
        return '\n - \t '.join([str(requirement) for requirement in self.requirements])