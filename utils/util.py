import os
from loaders.adult_loader import ADULT
from loaders.compas_loader import Compas
from loaders.credit_loader import Credit
from loaders.f1dp_loader import F1DPPreference
from functools import partial
import numpy as np
from multi_objective.utils_mo import model_from_name, method_from_name, dict_to_device
from multi_objective.objectives import get_objectives_from_name
from multi_objective.scores import get_scores_from_name


def make_directory(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def dataset_from_name(dataset, path, root,
                      sensible_groups,
                      ** kwargs):
    if dataset == 'adult':
        return ADULT(
            root=root,
            filename=path,
            sensible_groups=sensible_groups,
            **kwargs)
    elif dataset == 'compas':

        return Compas(
            root=root,
            filename=path,
            sensible_groups=sensible_groups,
            **kwargs
        )

    elif dataset == 'credit':
        return Credit(
            root=root,
            filename=path,
            sensible_groups=sensible_groups,
            **kwargs
        )
    elif dataset == 'f1dp_preference':
        return F1DPPreference(
            root=root,
            filename=path,
            sensible_groups=sensible_groups,
            **kwargs
        )
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def choose_model(pareto_front, ranking_fn):
    points = [point for point in pareto_front.get_pareto_front_list()]
    ranking = [ranking_fn(point)
               for point in pareto_front.get_pareto_front_list()]
    # print(points, ranking)
    # ranking = np.flip(np.sort(ranking))
    winner_idx = np.argmax(ranking)
    winner_model = pareto_front.get_model(winner_idx)
    winner_score = ranking[winner_idx]
    winner_point = points[winner_idx]
    ranking = np.flip(np.sort(ranking))
    return winner_model, winner_point, winner_score, ranking


def choose_model_epsilon_greedy(pareto_front, ranking_fn, epsilon, k):

    points = [(idx, point) for idx, point in
              enumerate(pareto_front.get_pareto_front_list())]

    ranking = [(idx, ranking_fn(point))
               for idx, point in points]
    """
    print(f'{pareto_front.get_pareto_front_list()}\n')
    for p, r in zip(points, ranking):
        print(f'{p[0]}: {p[1]} -> {r[1]}')
    print()
    """
    ranking = sorted(ranking, key=lambda p: p[1], reverse=True)
    n_scores = len(points[0][1])

    cpf = [r for r in ranking if r[1] == n_scores]
    upf = [r for r in ranking if r[1] < n_scores]

    proba = np.random.random()
    if (len(cpf) == 0):
        upf = sorted(upf, key=lambda p: p[1], reverse=True)
        if proba < epsilon:
            winner_idx = np.random.randint(0, min(k, len(upf)))
        else:
            winner_idx = upf[0]
    else:
        cpf = sorted(cpf, key=lambda p: p[1], reverse=True)
        if proba < epsilon:
            winner_idx = np.random.randint(0, min(k, len(cpf)))
        else:
            winner_idx = cpf[0]

    winner_model = pareto_front.get_model(
        ranking[winner_idx][0])
    winner_score = ranking[winner_idx][1]
    winner_point = points[ranking[winner_idx][0]]

    return winner_model, winner_point, winner_score, ranking


def _linear_combination_ranking(weights, scores):
    return weights[0]*scores[0] + (weights[1]*(1-scores[1]))


def _default_policy_ranking(weights, scores):
    final_score = 0
    if (scores[0] < 0.7) or (scores[1] > 0.6):
        final_score = -300
    else:
        final_score = weights[0]*(scores[0]*100 - 70)
        for w, score in zip(weights[1:], scores[1:]):
            final_score += w*(20 - (score*100))
        final_score = round(final_score, 3)
    return final_score


def _max_f1_ranking(weights, scores):
    return scores[0]


def _rank(thresholds, weights, scores):
    final_score = 0
    for s, threshold in zip(scores, thresholds):
        if not threshold(s):
            final_score = (s*100)-100
            return final_score
    if scores[1] <= 0.2:
        final_score = scores[0]*100
        return final_score
    else:
        final_score = scores[0] * 100
        for s, w in zip(scores[1:], weights[1:]):
            final_score -= (s-0.2) * 100
        return final_score


"""
def _default_policy_ranking_compas(weights, scores):
    final_score = 0
    if (scores[0] < 0.7) or (scores[1] > 0.7):
        final_score = -300
    else:
        f1 = (scores[0]*100 - 70)
        dp = (20 - scores[1]*100)
        final_score = round(weights[0]*f1 + weights[1]*dp, 3)
    return final_score
"""


def _default_policy_ranking_compas(weights, scores,
                                   thresholds=[0.7, 0.2],
                                   thresh_weights=[1.0, -1.0]):

    final_score = 0
    norm = np.linalg.norm(np.array(thresholds)-np.array(scores))
    for w, s, t, tw in zip(weights, scores, thresholds, thresh_weights):
        if s*tw >= t*tw:
            final_score += 1
        else:
            loss = abs(t-s)
            final_score -= loss
    return round(norm, 3)


def init_mo_method(student_config, model):
    score_names = []
    objective_names = []

    score_groups = []
    objective_groups = []
    scores_info = student_config.pop('scores')
    for name, group in scores_info:
        score_names.append(name)
        score_groups.append(group)

    objective_info = student_config.pop('objectives')

    for name, group in objective_info:
        objective_names.append(name)
        objective_groups.append(group)

    objectives = get_objectives_from_name(
        objective_names, objective_groups)
    scores = get_scores_from_name(
        score_names, score_groups)

    method = method_from_name(objectives=objectives,
                              model=model,
                              **student_config)

    weights = student_config['weights'][0]
    method.set_weights(weights)
    return {'method': method,
            'objectives': objectives,
            'scores': scores,
            'objectives_names': objective_names,
            'scores_names': score_names
            }


class Region:
    def __init__(self, f1_lim, fair_lim, score, compare_fn,
                 mo_weights=[0.2, 0.8], offset_idx=1):
        self.f1_lim = f1_lim
        self.fair_lim = fair_lim
        self.compare_fn = compare_fn
        self.score = score
        self.mo_weights = mo_weights
        self.offset_idx = offset_idx

    def belongs(self, point):
        f1 = point[0]
        fair = point[1]

        if self.f1_lim[0] <= f1 <= self.f1_lim[1]:
            if self.fair_lim[0] <= fair <= self.fair_lim[1]:
                return True
        return False

    def compare(self, p1, p2):
        return self.compare_fn(p1, p2)

    def compute_score(self, p):
        result = self.score
        if self.offset_idx == 0:
            result += p[self.offset_idx]
        elif self.offset_idx == 1:
            result += (1.0 - p[self.offset_idx])
        else:
            result = -p[0]

        return result

    def get_weights(self):
        return self.mo_weights


def _max_f1(p1, p2):
    if p1[0] >= p2[0]:
        return p1
    return p2


def _min_mse(p1, p2):
    if p1[0] <= p2[0]:
        return p1
    return p2


def _max_fair(p1, p2):
    if p1[0] <= p2[0]:
        return p1
    return p2


def choose_best_from_pf(pareto_front, domain, score_names):
    points = [(idx, point) for idx, point in
              enumerate(pareto_front.get_pareto_front_list())]

    winner_model_idx = 0
    winner_point = points[0][1]
    point_dict = {}
    for name, point in zip(score_names, winner_point):
        point_dict[name] = point

    winner_rank_score = domain(point_dict)

    for idx, point in points[1:]:
        point_dict = {}
        for name, p in zip(score_names, point):
            point_dict[name] = p
        score = domain(point_dict)
        if score > winner_rank_score:
            winner_rank_score = score
            winner_point = point
            winner_model_idx = idx

    winner_model = pareto_front.get_model(winner_model_idx)
    return winner_model, winner_point, winner_rank_score


class Domain:
    def __init__(self, regions, score_names) -> None:
        self.regions = regions
        self.score_names = score_names
        # print('Score names ', self.score_names)

    def compute_score(self, point_dict):
        # print(point_dict)
        point = [point_dict[name] for name in self.score_names]
        for region in self.regions:
            assert isinstance(region, Region)
            # print(point)
            if region.belongs(point):
                return region.compute_score(point), region
        raise KeyError(f'The point {point} does not belong to any region')


class F1DP_Domain:
    def __init__(self, score_names):
        r0 = Region([0, 0.6], [0, 0.5], 0, _max_f1,
                    mo_weights=[1, 0.01], offset_idx=0)

        r1 = Region([0, 1], [0.5, 1], 1, _max_fair,
                    mo_weights=[1, 3], offset_idx=1)

        r2 = Region([0.6, 0.7], [0.2, 0.5], 2, _max_fair,
                    mo_weights=[2, 1], offset_idx=1)
        r3 = Region([0.6, 0.7], [0., 0.2], 3, _max_f1,
                    mo_weights=[1, 0.1], offset_idx=0)

        r4 = Region([0.7, 0.75], [0.45, 0.5], 4, _max_fair,
                    mo_weights=[1, 2], offset_idx=1)
        r5 = Region([0.75, 0.80], [0.45, 0.5],  5, _max_fair,
                    mo_weights=[1, 3], offset_idx=1)
        r6 = Region([0.8, 1], [0.45, 0.5], 6, _max_fair,
                    mo_weights=[1, 3], offset_idx=1)

        r7 = Region([0.7, 0.75], [0.4, 0.45], 7, _max_fair,
                    mo_weights=[1, 2], offset_idx=1)
        r8 = Region([0.75, 0.80], [0.4, 0.45],  8, _max_fair,
                    mo_weights=[1, 2], offset_idx=1)
        r9 = Region([0.8, 1], [0.4, 0.45], 9, _max_fair,
                    mo_weights=[1, 2], offset_idx=1)

        r10 = Region([0.7, 0.75], [0.3, 0.4], 10, _max_fair,
                     mo_weights=[1, 2], offset_idx=1)
        r11 = Region([0.75, 0.8], [0.3, 0.4], 11, _max_fair,
                     mo_weights=[1, 2], offset_idx=1)
        r12 = Region([0.8, 1], [0.3, 0.4], 12, _max_fair,
                     mo_weights=[1, 2], offset_idx=1)

        r13 = Region([0.7, 0.75], [0.25, 0.3], 13, _max_fair,
                     mo_weights=[1, 1], offset_idx=1)
        r14 = Region([0.7, 0.75], [0.2, 0.25], 14, _max_fair,
                     mo_weights=[1, 1], offset_idx=1)
        r15 = Region([0.7, 0.75], [0, 0.2], 15, _max_f1,
                     mo_weights=[1, 0.01], offset_idx=0)

        r16 = Region([0.75, 0.8], [0.25, 0.3], 16, _max_fair,
                     mo_weights=[1, 1], offset_idx=1)
        r17 = Region([0.75, 0.8], [0.2, 0.25], 17, _max_fair,
                     mo_weights=[1, 1], offset_idx=1)
        r18 = Region([0.75, 0.8], [0, 0.2], 18, _max_f1,
                     mo_weights=[1, 0.01], offset_idx=0)

        r19 = Region([0.8, 1], [0.25, 0.3], 19, _max_fair,
                     mo_weights=[1, 1], offset_idx=1)
        r20 = Region([0.8, 1], [0.2, 0.25], 20, _max_fair,
                     mo_weights=[1, 1], offset_idx=1)
        r21 = Region([0.8, 1], [0, 0.2], 21, _max_f1,
                     mo_weights=[1, 0.01], offset_idx=0)

        regions = [r0, r1, r2, r3, r4, r5, r6, r7, r8,
                   r9, r10, r11, r12, r13, r14, r15,
                   r16, r17, r18, r19, r20, r21]
        self.domain = Domain(regions, score_names)
        self.maximum = 21

    def __call__(self, point_dict):
        return self.domain.compute_score(point_dict)[0]


class F1EOP_Domain:
    def __init__(self, score_names):
        r0 = Region([0, 0.6], [0, 0.5], 0, _max_f1,
                    offset_idx=0)

        r1 = Region([0, 1], [0.5, 1], 1, _max_fair,
                    offset_idx=1)

        r2 = Region([0.6, 0.7], [0.2, 0.5], 2, _max_fair,
                    offset_idx=1)
        r3 = Region([0.6, 0.7], [0., 0.2], 3, _max_f1,
                    offset_idx=0)

        r4 = Region([0.7, 1], [0.4, 0.5], 4, _max_fair,
                    offset_idx=1)

        r5 = Region([0.7, 1], [0.3, 0.4], 5, _max_fair,
                    offset_idx=1)

        r6 = Region([0.7, 0.75], [0.25, 0.3], 6, _max_fair,
                    offset_idx=1)
        r7 = Region([0.7, 0.75], [0.2, 0.25], 7, _max_fair,
                    offset_idx=1)
        r8 = Region([0.75, 0.8], [0.25, 0.3], 8, _max_fair,
                    offset_idx=1)
        r9 = Region([0.75, 0.8], [0.2, 0.25], 9, _max_fair,
                    offset_idx=1)
        r10 = Region([0.8, 1], [0.25, 0.3], 10, _max_fair,
                     offset_idx=1)
        r11 = Region([0.8, 1], [0.2, 0.25], 11, _max_fair,
                     offset_idx=1)
        r12 = Region([0.7, 0.75], [0.1, 0.2], 12, _max_fair,
                     offset_idx=1)
        r13 = Region([0.75, 0.8], [0.1, 0.2], 13, _max_fair,
                     offset_idx=1)
        r14 = Region([0.8, 1.0], [0.1, 0.2], 14, _max_fair,
                     offset_idx=1)
        r15 = Region([0.7, 0.75], [0, 0.1], 15, _max_f1,
                     offset_idx=0)
        r16 = Region([0.75, 0.8], [0, 0.1], 16, _max_f1,
                     offset_idx=0)
        r17 = Region([0.8, 1], [0, 0.1], 17, _max_f1,
                     offset_idx=0)

        regions = [r0, r1, r2, r3, r4, r5, r6, r7, r8,
                   r9, r10, r11, r12, r13, r14, r15,
                   r16, r17]
        self.domain = Domain(regions, score_names)
        self.maximum = 17

    def __call__(self, point_dict):
        return self.domain.compute_score(point_dict)[0]


class AndDomain:
    def __init__(self, domain_list):
        self.domain_list = domain_list

    def compute_score(self, point_dict):
        result = np.infty
        for domain in self.domain_list:
            result = min(result, domain(point_dict))
        return round(result, 3)

    def __call__(self, point_dict):
        return self.compute_score(point_dict)


class MulDomain:
    def __init__(self, domain_list):
        self.domain_list = domain_list

    def compute_score(self, point_dict):
        result = 1
        for domain in self.domain_list:
            result *= domain(point_dict)
        return round(result, 3)

    def __call__(self, point_dict):
        return self.compute_score(point_dict)


class OrDomain:
    def __init__(self, domain_list):
        self.domain_list = domain_list

    def compute_score(self, point_dict):
        result = 0
        for domain in self.domain_list:
            result = max(result, domain(point_dict))
        return round(result, 3)

    def __call__(self, point_dict):
        return self.compute_score(point_dict)


class Constraint:
    def __init__(self, constraint) -> None:
        self.constraint = constraint
        self.operator = self.constraint['op']
        self.threshold = self.constraint['thresh']
        self.quantity2check = self.constraint['quantity']

    def __call__(self, point):
        query_quantity = point[self.quantity2check]
        if self.operator == '>':
            return int(query_quantity > self.threshold)
        elif self.operator == '<':
            return int(query_quantity < self.threshold)
        elif self.operator == '=':
            return int(query_quantity == self.threshold)
        elif self.operator == '>=':
            return int(query_quantity >= self.threshold)
        elif self.operator == '<=':
            return int(query_quantity <= self.threshold)
        else:
            raise KeyError(f'The operator {self.operator} is unknown')


class ConstraintDomain:
    def __init__(self, constraints, domain):
        self.constraints = [Constraint(constraint)
                            for constraint in constraints]
        self.domain = domain

    def __call__(self, point_dict):
        score = 1
        score_domain = self.domain(point_dict)
        for constraint in self.constraints:
            if constraint.quantity2check == 'Score':
                score *= constraint({'Score': score_domain})
            else:
                score *= constraint(point_dict)
        return score*score_domain


class IfThenElseDomain:
    def __init__(self, condition, then_consequence, else_consequence) -> None:
        self.condition = condition
        self.then_consequence = then_consequence
        self.else_consequence = else_consequence

    def __call__(self, point_dict):
        premise_hold = self.condition(point_dict)
        if premise_hold > 0:
            score = self.then_consequence(point_dict)
        else:
            score = self.else_consequence(point_dict)
        return score


class WeightDomain:
    def __init__(self, weight, domain) -> None:
        self.weight = weight
        self.domain = domain

    def __call__(self, point_dict):
        score = self.domain(point_dict)*self.weight
        return round(score, 3)


def build_rule1():

    phi1 = F1DP_Domain(['F1:Group_Race', 'DDP:Group_Race'])
    phi2 = F1DP_Domain(['F1:Group_Race', 'DDP:Group_GR'])
    and1 = AndDomain([phi1, phi2])
    constraint1 = ConstraintDomain([{
        'quantity': 'DDP:Group_GR',
        'op': '<',
        'thresh': 0.4
    }],
        phi2)
    and2 = AndDomain([and1, constraint1])

    phi3 = F1EOP_Domain(['F1:Group_Race', 'DEO:Group_Race'])
    phi4 = F1EOP_Domain(['F1:Group_Race', 'DEO:Group_GR'])
    constraint2 = ConstraintDomain([{
        'quantity': 'DEO:Group_GR',
        'op': '<',
        'thresh': 0.3
    }],
        phi4)
    and3 = AndDomain([phi3, phi4])
    and4 = AndDomain([and3, constraint2])
    return OrDomain([and2, and4])


def build_rule2():
    phi1 = F1DP_Domain(['F1:Group_Race', 'DDP:Group_Gender'])
    c = {
        'quantity': 'DDP:Group_Gender',
        'op': '<',
        'thresh': 0.2
    }
    constraint = ConstraintDomain([c], phi1)
    return AndDomain([phi1, constraint])


def build_rule3():
    phi1 = F1DP_Domain(['F1:Group_Race', 'DDP:Group_Race'])
    c = {
        'quantity': 'DDP:Group_Race',
        'op': '<',
        'thresh': 0.2
    }
    constraint = ConstraintDomain([c], phi1)
    return AndDomain([phi1, constraint])


def build_rule4():
    phi1 = F1DP_Domain(['F1:Group_Race', 'DDP:Group_GR'])
    c = {
        'quantity': 'DDP:Group_GR',
        'op': '<',
        'thresh': 0.2
    }
    constraint = ConstraintDomain([c], phi1)
    return AndDomain([phi1, constraint])


def build_rule5():
    phi1 = F1EOP_Domain(['F1:Group_Race', 'DEO:Group_GR'])
    c = {
        'quantity': 'DEO:Group_GR',
        'op': '<=',
        'thresh': 0.1
    }
    constraint = ConstraintDomain([c], phi1)
    return AndDomain([phi1, constraint])


def build_rule_6():
    phi1 = build_rule4()
    phi2 = build_rule5()
    return OrDomain([phi1, phi2])


def build_rule7():
    phi1 = F1DP_Domain(['F1:Group_Race', 'DDP:Group_Gender'])
    c = {
        'quantity': 'DDP:Group_Gender',
        'op': '<',
        'thresh': 0.3
    }
    constraint = ConstraintDomain([c], phi1)
    return AndDomain([phi1, constraint])


def build_rule8():
    phi1 = F1DP_Domain(['F1:Group_Race', 'DDP:Group_Race'])
    c = {
        'quantity': 'DDP:Group_Race',
        'op': '<',
        'thresh': 0.3
    }
    constraint = ConstraintDomain([c], phi1)
    return AndDomain([phi1, constraint])


def build_rule9():
    phi1 = F1DP_Domain(['F1:Group_Race', 'DDP:Group_GR'])
    c = {
        'quantity': 'DDP:Group_GR',
        'op': '<',
        'thresh': 0.3
    }
    constraint = ConstraintDomain([c], phi1)
    return AndDomain([phi1, constraint])


def build_req_a():
    phi1 = build_rule2()
    phi2 = build_rule3()
    phi3 = build_rule_6()
    req = WeightDomain(5, AndDomain([phi1, phi2, phi3]))
    return req


def build_req_b():
    phi1 = build_rule2()
    phi2 = build_rule3()
    phi3 = build_rule_6()

    part1 = AndDomain([phi1, phi2])
    part2 = AndDomain([phi1, phi3])
    part3 = AndDomain([phi2, phi3])
    req = WeightDomain(3, OrDomain([part1, part2, part3]))
    return req


def build_req_c():
    phi1 = build_rule2()
    phi2 = build_rule3()
    phi3 = build_rule_6()
    req = WeightDomain(2, OrDomain([phi1, phi2, phi3]))
    return req


def build_req_d():
    phi1 = build_rule7()
    phi2 = build_rule8()
    phi3 = build_rule9()
    req = WeightDomain(1, OrDomain([phi1, phi2, phi3]))
    return req


def build_req():
    r1 = build_req_a()
    r2 = build_req_b()
    r3 = build_req_c()
    r4 = build_req_d()
    c1 = ConstraintDomain([
        {
            'quantity': 'Score',
            'op': '>',
            'thresh': 0
        }
    ], r1)

    c2 = ConstraintDomain([
        {
            'quantity': 'Score',
            'op': '>',
            'thresh': 0
        }
    ], r2)
    c3 = ConstraintDomain([
        {
            'quantity': 'Score',
            'op': '>',
            'thresh': 0
        }
    ], r3)
    req = IfThenElseDomain(c1, r1,
                           IfThenElseDomain(
                               c2, r2,
                               IfThenElseDomain(
                                   c3, r3, r4
                               )
                           ))
    return req


def build_rule2_simple():
    phi1 = F1DP_Domain(['F1:Group_Race', 'DEODDS:Group_GR'])
    return phi1


def build_rule3_simple():
    phi1 = F1DP_Domain(['F1:Group_Race', 'DDP:Group_GR'])
    return phi1


def build_req_simple():
    phi1 = build_rule2_simple()
    phi2 = build_rule3_simple()
    req = MulDomain([phi1, phi2])
    return req


def build_scenario3_rules():
    phi1 = F1DP_Domain(['F1:Group_Race', 'DDP:Group_GR'])
    phi2 = F1EOP_Domain(['F1:Group_Race', 'DEO:Group_GR'])
    phi3 = F1EOP_Domain(['F1:Group_Race', 'DPE:Group_GR'])
    return AndDomain([phi1, phi2, phi3])


class MultiGroupDomain:
    def __init__(self, domain_list):
        self.domain_list = domain_list

    def compute_score(self, point_dict):
        result = np.mean([d(point_dict)for d in self.domain_list])
        return round(result, 3)

    def __call__(self, point_dict):
        return self.compute_score(point_dict)


class AvgDomain:
    def __init__(self, domain_list):
        self.domain_list = domain_list

    def compute_score(self, point_dict):
        result = np.mean([d(point_dict)for d in self.domain_list])
        return round(result, 3)

    def __call__(self, point_dict):
        return self.compute_score(point_dict)


class SumDomain:
    def __init__(self, domain_list):
        self.domain_list = domain_list

    def compute_score(self, point_dict):
        result = np.sum([d(point_dict)for d in self.domain_list])
        return round(result, 3)

    def __call__(self, point_dict):
        return self.compute_score(point_dict)


def build_multi_group_rule():
    domains = []
    for i in range(6):
        domains.append(F1DP_Domain(
            ['F1:Group_0', f'DDP:Group_{i}']
        ))
    return MultiGroupDomain(domains)


def build_adult3_rule():
    dp_domains = []
    eo_domains = []
    dp_domains.append(F1DP_Domain(
        ['F1:Group_Race', 'DDP:Group_Gender']
    ))
    dp_domains.append(F1DP_Domain(
        ['F1:Group_Race', 'DDP:Group_Race']
    ))
    dp_domains.append(F1DP_Domain(
        ['F1:Group_Race', 'DDP:Group_Married']
    ))
    phi_dp = AvgDomain(dp_domains)

    eo_domains.append(F1EOP_Domain(
        ['F1:Group_Race', 'DEO:Group_Gender']
    ))
    eo_domains.append(F1EOP_Domain(
        ['F1:Group_Race', 'DEO:Group_Race']
    ))
    eo_domains.append(F1DP_Domain(
        ['F1:Group_Race', 'DEO:Group_Married']
    ))
    phi_eo = AvgDomain(eo_domains)
    return SumDomain([phi_dp, phi_eo])


def build_credit2_rules():
    dp_domains = []
    dp_domains.append(F1DP_Domain(
        ['F1:Group_Race', 'DEO:Group_Gender']
    ))
    dp_domains.append(F1DP_Domain(
        ['F1:Group_Race', 'DEO:Group_Race']
    ))
    return AndDomain(dp_domains)


def build_adult_contrasting_rule():
    dp_domains = []
    dp_domains.append(F1DP_Domain(
        ['F1:Group_Race', 'DDP:Group_Race']
    ))
    dp_domains.append(F1DP_Domain(
        ['F1:Group_Race', 'DEO:Group_Race']
    ))
    return AndDomain(dp_domains)


def build_domain(name, score_names=[]):
    if name == 'F1DP':
        return F1DP_Domain(['F1:Group_Gender', 'DDP:Group_Gender'])
    elif name == 'F1EO':
        return F1DP_Domain(['F1:Group_Race', 'DEO:Group_Race'])
    elif name == 'rule_1':
        return build_rule1()
    elif name == 'rule_2':
        return build_rule2()
    elif name == 'rule_3':
        return build_rule3()
    elif name == 'rule_4':
        return build_rule4()
    elif name == 'req':
        return build_req()
    elif name == 'max_f1':
        return MaxF1_Domain()
    elif name == 'min_mse':
        return MinMSE_Domain()
    elif name == 'req_simple':
        return build_req_simple()
    elif name == 'scenario3':
        return build_scenario3_rules()
    elif name == 'multi':
        return build_multi_group_rule()
    elif name == 'adult3':
        return build_adult3_rule()
    elif name == 'credit2':
        return build_credit2_rules()
    elif name == 'adult_contrasting':
        return build_adult_contrasting_rule()
    return


def ranking_function_from_name(name):
    if name == 'linear_combination':
        return partial(_linear_combination_ranking)
    elif name == 'default':
        return partial(_default_policy_ranking)
    elif name == 'max_f1':
        return partial(_max_f1_ranking)
    elif name == 'default_compas':
        return partial(_default_policy_ranking_compas)
    elif name == 'rank':

        def below_thresh(thresh, value):
            return value <= thresh

        def above_thresh(thresh, value):
            return value >= thresh
        thresholds = [partial(above_thresh, 0.6), partial(above_thresh, 0)]
        return partial(_rank, thresholds)
    else:
        raise ValueError("Unknown ranking function")


class MaxF1_Domain:
    def __init__(self, score_names=['F1:Group_Race', 'F1:Group_Race']):
        r0 = Region([0, 1], [0, 1], 1, _max_f1,
                    offset_idx=0)
        regions = [r0]
        self.domain = Domain(regions, score_names)
        self.maximum = 2

    def __call__(self, point_dict):

        return self.domain.compute_score(point_dict)[0]


class MinMSE_Domain:
    def __init__(self, score_names=['L2Distance:Group_Gender', 'L2Distance:Group_Gender']):
        r0 = Region([0, np.infty], [0, np.infty], 1, _min_mse,
                    offset_idx=2)
        regions = [r0]
        self.domain = Domain(regions, score_names)
        self.maximum = 2

    def __call__(self, point_dict):

        return self.domain.compute_score(point_dict)[0]
