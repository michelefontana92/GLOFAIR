from experiments import GlofairExperiment
import shutil
from .adult_run import AdultRun
from ..run_factory import register_run
from requirements import RequirementSet,ConstrainedRequirement,UnconstrainedRequirement
from surrogates import SurrogateFunctionSet,SurrogateFactory
from metrics import MetricsFactory

@register_run('adult_glofair_dp_eod_race')
class AdultGlofairDPEODRaceRun(AdultRun):
    def __init__(self,**kwargs) -> None:
        super(AdultGlofairDPEODRaceRun, self).__init__(**kwargs)
        self.project_name = 'AdultGlofairDPEODRace'
        self.project_name =  kwargs.get('project_name')
        self.start_index = kwargs.get('start_index')
        self.num_clients = 10
        self.lr=1e-4
        self.num_federated_rounds = 100
        self.training_group_name = 'Race'

        self.surrogate_set = SurrogateFunctionSet([SurrogateFactory.create(name='demographic_parity',
                                                               group_name=self.training_group_name,
                                                               unique_group_ids={
                                                                   self.training_group_name:list(range(2))
                                                                   },
                                                               reduction='mean',
                                                               weight=2
                                                               ),

                                      SurrogateFactory.create(name='performance',
                                                              surrogate_weight=1)
                                      ])

        self.requirement_set = RequirementSet([
            UnconstrainedRequirement(name='unconstraned_performance_requirement',
                             metric = MetricsFactory.create_metric(
                                    metric_name='performance'),
                             weight=1,
                             mode='max',
                             bound=1.0,
                             performance_metric='f1'
                             ),
                   

                    ConstrainedRequirement(name='dp_requirement',
                                           metric = MetricsFactory.create_metric(
                                                    metric_name='demographic_parity',
                                                    group_name=self.training_group_name,
                                                    group_ids={self.training_group_name:list(range(2))}),
                                            weight=3,
                                            operator='<=',
                                            threshold=0.2),
                    ConstrainedRequirement(name='eod_requirement',
                                           metric = MetricsFactory.create_metric(
                                                    metric_name='equalized_odds',
                                                    group_name=self.training_group_name,
                                                    group_ids={self.training_group_name:list(range(2))}),
                                            weight=3,
                                            operator='<=',
                                            threshold=0.2),
                        ])
    
    def setUp(self):
       
        self.experiment = GlofairExperiment( sensitive_attributes=self.sensitive_attributes,
                                            dataset=self.dataset,
                                            data_root=self.data_root,
                                            model=self.model,
                                            num_clients=self.num_clients,
                                            num_federated_rounds=self.num_federated_rounds,
                                            lr=self.lr,
                                            project=self.project_name,
                                            training_group_name=self.training_group_name,
                                            surrogate_set=self.surrogate_set,
                                            requirement_set=self.requirement_set,
                                            start_index=self.start_index
                                            )

    def run(self):
        self.experiment.setup()
        self.experiment.run()

    def tearDown(self) -> None:
        shutil.rmtree(f'checkpoints/{self.project_name}')