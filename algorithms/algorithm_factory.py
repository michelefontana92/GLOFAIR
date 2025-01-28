_ALGORITHM = {}
def register_algorithm(algorithm_name):
    def decorator(fn):
        _ALGORITHM[algorithm_name] = fn
        return fn
    return decorator


class AlgorithmFactory:
    @staticmethod
    def create(name,**kwargs):
        if name not in _ALGORITHM:
            raise ValueError(f"Algorithm {name} not found")
        return _ALGORITHM[name](**kwargs)