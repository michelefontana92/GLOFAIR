from torchmetrics import StatScores
import torch 
from abc import ABC, abstractmethod
from .metrics_factory import register_metric
from .base_metric import BaseMetric
from surrogates import SurrogateFactory,WassersteinDemographicParitySurrogate,WassersteinEqualOpportunitySurrogate,WassersteinEqualizedOddsSurrogate

@register_metric('statistic_scores')
class StatisticScores(BaseMetric):
    def __init__(self,**kwargs):
        task = kwargs.get('task','multiclass')
        num_classes = kwargs.get('num_classes',2)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.stat_scores = StatScores(task=task,
                                      num_classes=num_classes).to(self.device)
    
    def calculate(self, y_pred, y_true):
        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)
        self.stat_scores.update(y_pred, y_true)
    
    def get(self,normalize=False):
        support = self.stat_scores.tp + self.stat_scores.fp + self.stat_scores.tn + self.stat_scores.fn
        stats = {"tp": self.stat_scores.tp, 
            "fp": self.stat_scores.fp,
            "tn": self.stat_scores.tn,
            "fn": self.stat_scores.fn,
            "tpr": self.stat_scores.tp / (self.stat_scores.tp + self.stat_scores.fn) if (self.stat_scores.tp + self.stat_scores.fn) != 0 else 0,
            "fpr": self.stat_scores.fp / (self.stat_scores.fp + self.stat_scores.tn) if (self.stat_scores.fp + self.stat_scores.tn) != 0 else 0,
            "tnr": self.stat_scores.tn / (self.stat_scores.tn + self.stat_scores.fp) if (self.stat_scores.tn + self.stat_scores.fp) != 0 else 0,
            "fnr": self.stat_scores.fn / (self.stat_scores.fn + self.stat_scores.tp) if (self.stat_scores.fn + self.stat_scores.tp) != 0 else 0,
            "base_rate": (self.stat_scores.tp + self.stat_scores.fp) / support if support != 0 else 0,
            }
            
        if normalize:
            stats["tp"] = self.stat_scores.tp / support
            stats["fp"] = self.stat_scores.fp / support
            stats["tn"] = self.stat_scores.tn / support
            stats["fn"] = self.stat_scores.fn / support
        return stats    
       
    def reset(self):
        self.stat_scores.reset()



class GroupFairnessMetric(BaseMetric):
    def __init__(self,**kwargs):
        task = kwargs.get('task','binary')
        self.num_classes = kwargs.get('num_classes',2)
        group_ids = kwargs.get('group_ids')
        self.group_ids = group_ids
        self.group_name = kwargs.get('group_name')
        assert isinstance(group_ids, dict), "group_ids must be a dictionary"
        assert len(group_ids[self.group_name]) > 1, "group_ids must have at least 2 groups"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_multiclass = kwargs.get('use_multiclass',False)
        _REDUCTION_TYPES = {
            'min': torch.min, 
            'mean': torch.mean, 
            'max': torch.max
        }

        self._reduction = _REDUCTION_TYPES.get(kwargs.get('reduction','max'))
    
        self.stats_per_group = {}
        for group_id in group_ids[self.group_name]:
            self.stats_per_group[group_id] = StatisticScores(task=task,
                                                             num_classes=2)
        self.stats_per_class = {}
        for current_class in range(self.num_classes):
            self.stats_per_class[current_class] = {}
            for group_id in group_ids[self.group_name]:
                self.stats_per_class[current_class][group_id] = StatisticScores(task=task,
                                                             num_classes=2)

    def calculate(self, y_pred, y_true, group_ids:dict):
        current_group_ids:list = group_ids[self.group_name]
        #print('Current group ids: ',current_group_ids)
        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)
        assert len(y_pred) == len(y_true) == len(current_group_ids), "y_pred, y_true and group_ids must have the same length"
        
        if not self.use_multiclass:
            
            for group_id in torch.unique(current_group_ids):
                if group_id != -1:
                    #print('y_pred:',y_pred)
                    #print('group_ids:',group_ids[self.group_name])
                    y_pred_group = y_pred[current_group_ids==group_id.item()]
                    y_true_group = y_true[current_group_ids==group_id.item()]
                    self.stats_per_group[group_id.item()].calculate(y_pred_group, y_true_group)
        else:
            for current_class in range(self.num_classes):
                y_pred_class = torch.where(y_pred==current_class,1,0)
                y_true_class = torch.where(y_true==current_class,1,0)
                """
                print('Current class: ',current_class)
                print('y_true: ',y_true[:5])
                print('y_true_class: ',y_true_class[:5])
                print('y_pred: ',y_pred[:5])
                print('y_pred_class: ',y_pred_class[:5])
                print()
                """
                for group_id in torch.unique(current_group_ids):
                    if group_id != -1:
                        y_pred_group = y_pred_class[current_group_ids==group_id.item()]
                        y_true_group = y_true_class[current_group_ids==group_id.item()]
                        self.stats_per_class[current_class][group_id.item()].calculate(y_pred_group, y_true_group)
    
    def get(self,normalize=False):
       pass

    def get_stats_per_group(self,group_id):     
        pass
             
    def reset(self):
        for _,stats in self.stats_per_group.items():
            stats.reset()
        for _,stats_dict in self.stats_per_class.items():
            for _,stats in stats_dict.items():
                stats.reset()


@register_metric('demographic_parity')
class DemographicParity(GroupFairnessMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stats_per_group_diff = []
        self.stats_per_class_group_diff = {}
        self.metrics_per_class = []
        for current_class in range(self.num_classes):
            self.stats_per_class_group_diff[current_class] = []

        self.wasserstein_distance:WassersteinDemographicParitySurrogate = SurrogateFactory.create(
                                                              'wasserstein_demographic_parity',
                                                               group_name=self.group_name,
                                                               unique_group_ids=self.group_ids,
                                                               reduction='mean',
                                                               weight=1)
    
       
    def get(self):
        if not self.use_multiclass:
            group_ids = list(self.stats_per_group.keys())
            for i in range(len(group_ids)):
                for j in range(i+1,len(group_ids)):
                    self.stats_per_group_diff.append(
                        abs(self.stats_per_group[group_ids[i]].get()['base_rate'] - self.stats_per_group[group_ids[j]].get()['base_rate']))
                    
            
            return {
                    f'demographic_parity_{self.group_name}':self._reduction(
                        torch.tensor(self.stats_per_group_diff))
                    }
        else: 
            group_ids = list(self.stats_per_class[0].keys())
            for current_class in range(self.num_classes):
                for i in range(len(group_ids)):
                    for j in range(i+1,len(group_ids)):
                        self.stats_per_class_group_diff[current_class].append(
                            abs(self.stats_per_class[current_class][group_ids[i]].get()['base_rate'] - self.stats_per_class[current_class][group_ids[j]].get()['base_rate']))

                dp = self._reduction(torch.tensor(self.stats_per_class_group_diff[current_class]))
                self.metrics_per_class.append(dp)
            """
            print('Attribute: ',self.group_name)
            for i in range(len(group_ids)):
                print('Group: ',group_ids[i])
                for c in range(self.num_classes):
                    br = self.stats_per_class[c][group_ids[i]].get()["base_rate"]
                    print(f'Class {c}: base rate: {br}')
            print()      
            """
            return {
                    f'demographic_parity_{self.group_name}':self._reduction(
                        torch.tensor(self.metrics_per_class))
                    }
    def get_stats_per_group(self, group_id):
        return self.stats_per_group[group_id].get()['base_rate'][0].item()
            
    def reset(self):
        super().reset()
        self.stats_per_group_diff = []
        self.metrics_per_class = []
        for current_class in range(self.num_classes):
            self.stats_per_class_group_diff[current_class] = []
    
    def compute_wasserstein_distance(self,logits,labels,group_ids,**kwargs):
        distance = self.wasserstein_distance(logits=logits,
                                             labels=labels,
                                             group_ids=group_ids,
                                             **kwargs)
        return {f'wasserstein_distance_demographic_parity_{self.group_name}':
                distance
                }
    
@register_metric('equal_opportunity')
class EqualOpportunity(GroupFairnessMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stats_per_group_diff = []
        self.stats_per_class_group_diff = {}
        self.metrics_per_class = []
        for current_class in range(self.num_classes):
            self.stats_per_class_group_diff[current_class] = []
        self.wasserstein_distance:WassersteinEqualOpportunitySurrogate = SurrogateFactory.create(
                                                              'wasserstein_equal_opportunity',
                                                               group_name=self.group_name,
                                                               unique_group_ids=self.group_ids,
                                                               reduction='mean',
                                                               weight=1)
    def get(self):
        if not self.use_multiclass: 
            group_ids = list(self.stats_per_group.keys())
            for i in range(len(group_ids)):
                for j in range(i+1,len(group_ids)):
                    self.stats_per_group_diff.append(abs(self.stats_per_group[group_ids[i]].get()['tpr'] - self.stats_per_group[group_ids[j]].get()['tpr']))
        
            return {f'equal_opportunity_{self.group_name}':
                    self._reduction(
                        torch.tensor(self.stats_per_group_diff))
                    }
        else: 
            group_ids = list(self.stats_per_class[0].keys())
            for current_class in range(self.num_classes):
                for i in range(len(group_ids)):
                    for j in range(i+1,len(group_ids)):
                        self.stats_per_class_group_diff[current_class].append(
                            abs(self.stats_per_class[current_class][group_ids[i]].get()['tpr'] - self.stats_per_class[current_class][group_ids[j]].get()['tpr']))
    
                eo = self._reduction(torch.tensor(self.stats_per_class_group_diff[current_class]))
                self.metrics_per_class.append(eo)
            return {
                    f'equal_opportunity_{self.group_name}':self._reduction(
                        torch.tensor(self.metrics_per_class))
                    }
    def get_stats_per_group(self, group_id):
        return torch.tensor(
            self.stats_per_group[group_id].get()['tpr'][0].item()
            )
    def reset(self):
        super().reset()
        self.stats_per_group_diff = []
        self.metrics_per_class = []
        for current_class in range(self.num_classes):
            self.stats_per_class_group_diff[current_class] = []
    
    def compute_wasserstein_distance(self,logits,labels,group_ids,**kwargs):
        distance:WassersteinEqualOpportunitySurrogate = self.wasserstein_distance(logits=logits,
                                             labels=labels,
                                             group_ids=group_ids,
                                             **kwargs)
        return {f'wasserstein_distance_equal_opportunity_{self.group_name}':
                distance
                }
    
@register_metric('equalized_odds')
class EqualizedOdds(GroupFairnessMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stats_per_group_diff_tpr = []
        self.stats_per_group_diff_fpr = []
        self.stats_per_class_group_diff_tpr = {}
        self.stats_per_class_group_diff_fpr = {}
        self.metrics_per_class = []
        for current_class in range(self.num_classes):
            self.stats_per_class_group_diff_tpr[current_class] = []
            self.stats_per_class_group_diff_fpr[current_class] = []
        self.wasserstein_distance:WassersteinEqualizedOddsSurrogate = SurrogateFactory.create(
                                                              'wasserstein_equalized_odds',
                                                               group_name=self.group_name,
                                                               unique_group_ids=self.group_ids,
                                                               reduction='mean',
                                                               weight=1)
    def get(self):
        if not self.use_multiclass:
            group_ids = list(self.stats_per_group.keys()) 
            for i in range(len(group_ids)):
                for j in range(i+1,len(group_ids)):
                    self.stats_per_group_diff_tpr.append(abs(self.stats_per_group[group_ids[i]].get()['tpr']-self.stats_per_group[group_ids[j]].get()['tpr']))
                    self.stats_per_group_diff_fpr.append(abs(self.stats_per_group[group_ids[i]].get()['fpr']-self.stats_per_group[group_ids[j]].get()['fpr']))
            
            return {
                f'equalized_odds_{self.group_name}':
                torch.max(
                self._reduction(
                    torch.tensor(self.stats_per_group_diff_tpr)),
                self._reduction(
                    torch.tensor(self.stats_per_group_diff_fpr))
            )
            }
        else:
            group_ids = list(self.stats_per_class[0].keys())
            for current_class in range(self.num_classes):
                for i in range(len(group_ids)):
                    for j in range(i+1,len(group_ids)):
                        self.stats_per_class_group_diff_tpr[current_class].append(
                            abs(self.stats_per_class[current_class][group_ids[i]].get()['tpr'] - self.stats_per_class[current_class][group_ids[j]].get()['tpr']))
                        self.stats_per_class_group_diff_fpr[current_class].append(
                            abs(self.stats_per_class[current_class][group_ids[i]].get()['fpr'] - self.stats_per_class[current_class][group_ids[j]].get()['fpr']))
                eo = torch.max(self._reduction(torch.tensor(self.stats_per_class_group_diff_tpr[current_class])),
                               self._reduction(torch.tensor(self.stats_per_class_group_diff_fpr[current_class])))
                self.metrics_per_class.append(eo)
            return {
                    f'equalized_odds_{self.group_name}':self._reduction(
                        torch.tensor(self.metrics_per_class))
                    }
    def reset(self):
        super().reset()
        self.stats_per_group_diff_tpr = []
        self.metrics_per_class = []
        self.stats_per_group_diff_fpr = []
        for current_class in range(self.num_classes):
            self.stats_per_class_group_diff_tpr[current_class] = []
            self.stats_per_class_group_diff_fpr[current_class] = []
    def compute_wasserstein_distance(self,logits,labels,group_ids,**kwargs):
        distance:WassersteinEqualizedOddsSurrogate = self.wasserstein_distance(logits=logits,
                                             labels=labels,
                                             group_ids=group_ids,
                                             **kwargs)
        return {f'wasserstein_distance_equalized_odds_{self.group_name}':
                distance
                }

@register_metric('wasserstein_distance')
class WassersteinDistance(BaseMetric):
    def __init__(self, **kwargs):
        group_ids = kwargs.get('group_ids')
        self.group_name = kwargs.get('group_name')
        self.metric_name = kwargs.get('metric_name')
        assert isinstance(group_ids, dict), "group_ids must be a dictionary"
        assert len(group_ids[self.group_name]) > 1, "group_ids must have at least 2 groups"

        _REDUCTION_TYPES = {
            'min': torch.min, 
            'mean': torch.mean, 
            'max': torch.max
        }

        self._reduction = _REDUCTION_TYPES.get(kwargs.get('reduction','mean'))
        self.distances_per_group = []
          
    def get(self): 
        group_ids = list(self.stats_per_group.keys())
        for i in range(len(group_ids)):
            for j in range(i+1,len(group_ids)):
                self.stats_per_group_diff.append(abs(self.stats_per_group[group_ids[i]].get()['tpr'] - self.stats_per_group[group_ids[j]].get()['tpr']))
    
        return {f'wasserstein_distance_{self.group_name}':
                self._reduction(
                    torch.tensor(self.stats_per_group_diff))
        }
    
    def calculate(self, probabilities, labels, group_ids:dict):
        current_group_ids:list = group_ids[self.group_name]
        assert len(probabilities) == len(labels) == len(current_group_ids), "y_pred, y_true and group_ids must have the same length"
        for group_id in torch.unique(current_group_ids):
            if group_id != -1:
                y_pred_group = y_pred[current_group_ids==group_id.item()]
                y_true_group = y_true[current_group_ids==group_id.item()]
                self.stats_per_group[group_id.item()]= wasserstein_distance_threshold(y_pred_group, y_true_group)
    
    def reset(self):
        super().reset()
        self.distances_per_group = []
