_METRICS={}

def register_metric(metric_name):
    def decorator(fn):
        _METRICS[metric_name] = fn
        return fn
    return decorator

class MetricsFactory:
    @staticmethod
    def create_metric(metric_name, **kwargs):
        if metric_name not in _METRICS:
            raise ValueError(f"Unknown metric: {metric_name}")
        return _METRICS[metric_name](**kwargs)
    
    @staticmethod
    def serialize(metric_instance):
        """
        Serializza un'istanza di metrica.
        """
        metric_type = type(metric_instance).__name__
        metric_name = next(
            (name for name, cls in _METRICS.items() if isinstance(metric_instance, cls)),
            None
        )
        if metric_name is None:
            raise ValueError(f"Metric {metric_type} non registrata.")
        return {
            'type': metric_name,
            'params': metric_instance.__dict__  # Salva lo stato interno della metrica
        }

    @staticmethod
    def deserialize(data):
        """
        Ricostruisce un'istanza di metrica dai dati serializzati.
        """
        metric_name = data['type']
        if metric_name not in _METRICS:
            raise ValueError(f"Metric {metric_name} non registrata.")
        metric_cls = _METRICS[metric_name]
        instance = metric_cls.__new__(metric_cls)  # Crea l'istanza senza chiamare __init__
        instance.__dict__.update(data['params'])  # Ripristina lo stato interno
        return instance