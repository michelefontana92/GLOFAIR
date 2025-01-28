_AGGREGATORS = {}

def register_aggregator(name:str):
    def decorator(cls):
        if name in _AGGREGATORS:
            raise ValueError(f"Cannot register {name} as "
                             f"it is already registered")
        _AGGREGATORS[name] = cls
        return cls
    return decorator

class AggregatorFactory:
    @staticmethod
    def create(name:str,**kwargs):
        if name not in _AGGREGATORS:
            raise ValueError(f"{name} not found")
        return _AGGREGATORS[name](**kwargs)