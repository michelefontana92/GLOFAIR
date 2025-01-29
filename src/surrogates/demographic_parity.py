from .base_surrogate import BaseSurrogate,BaseBinarySurrogate,BaseBinaryWassersteinSurrogate
from .surrogate_factory import register_surrogate


class BinaryDemographicParitySurrogate(BaseBinarySurrogate):
    def __init__(self, **kwargs) -> None:
        super(BinaryDemographicParitySurrogate, self).__init__(**kwargs)
        self.positive_group_id:int = kwargs.get('positive_group_id')
        self.negative_group_id:int = kwargs.get('negative_group_id')
        assert self.positive_group_id != self.negative_group_id, f'positive_group_id and negative_group_id should be different'
        assert self.positive_group_id is not None, f'positive_group_id should not be None'
        assert self.negative_group_id is not None, f'negative_group_id should not be None'

    def _compute_statistic(self, logits, labels, group_ids):
        #print('positive_group_id:',self.positive_group_id)
        #print('negative_group_id:',self.negative_group_id)
        positive_mask = (group_ids[self.group_name] == self.positive_group_id).squeeze()
        negative_mask = (group_ids[self.group_name] == self.negative_group_id).squeeze()
        surrogate = self._calculate(logits, positive_mask, negative_mask)
        return surrogate
    

class BinaryWassersteinDemographicParitySurrogate(BaseBinaryWassersteinSurrogate):
    def __init__(self, **kwargs) -> None:
        super(BinaryWassersteinDemographicParitySurrogate, self).__init__(**kwargs)
        self.positive_group_id:int = kwargs.get('positive_group_id')
        self.negative_group_id:int = kwargs.get('negative_group_id')
        assert self.positive_group_id != self.negative_group_id, f'positive_group_id and negative_group_id should be different'
        assert self.positive_group_id is not None, f'positive_group_id should not be None'
        assert self.negative_group_id is not None, f'negative_group_id should not be None'

    def _compute_statistic(self, logits, labels, group_ids,**kwargs):
        positive_mask = (group_ids[self.group_name] == self.positive_group_id).squeeze()
        negative_mask = (group_ids[self.group_name] == self.negative_group_id).squeeze()
        surrogate = self._calculate(logits, positive_mask, negative_mask,**kwargs)
        return surrogate    
   
@register_surrogate('demographic_parity')
class DemographicParitySurrogate(BaseSurrogate):
    def __init__(self,**kwargs) -> None:
        super(DemographicParitySurrogate, self).__init__(**kwargs)
        self._init_surrogates(BinaryDemographicParitySurrogate)

@register_surrogate('wasserstein_demographic_parity')
class WassersteinDemographicParitySurrogate(BaseSurrogate):
    def __init__(self,**kwargs) -> None:
        super(WassersteinDemographicParitySurrogate, self).__init__(**kwargs)
        self._init_surrogates(BinaryWassersteinDemographicParitySurrogate)
  