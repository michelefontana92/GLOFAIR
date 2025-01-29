from .surrogate_factory import SurrogateFactory
from .demographic_parity import DemographicParitySurrogate,WassersteinDemographicParitySurrogate
from .equal_opportunity import EqualOpportunitySurrogate,WassersteinEqualOpportunitySurrogate
from .predictive_equality import PredictiveEqualitySurrogate,WassersteinPredictiveEqualitySurrogate
from .equalized_odds import EqualizedOddsSurrogate,WassersteinEqualizedOddsSurrogate
from .performance import PerformanceSurrogate
from .base_surrogate import BaseSurrogate,BaseBinarySurrogate
from .surrogate_set import SurrogateFunctionSet
from .differentiable_performance import *
from .differentiable_fairness import *
from .wasserstein import *

__all__ = ['SurrogateFactory','BaseSurrogate','BaseBinarySurrogate',
           'PerformanceSurrogate','SurrogateFunctionSet',
           'WassersteinDemographicParitySurrogate',
           'WassersteinEqualOpportunitySurrogate',
           'WassersteinPredictiveEqualitySurrogate',
           'WassersteinEqualizedOddsSurrogate',
           'BinaryAccuracySurrogate',
           'BinaryPrecisionSurrogate',
           'BinaryRecallSurrogate',
           'BinaryF1Surrogate',
           'DifferentiableDemographicParitySurrogate']