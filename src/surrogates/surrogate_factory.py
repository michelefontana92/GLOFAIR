_SURROGATES = {}
def register_surrogate(surrogate_name):
    def decorator(fn):
        _SURROGATES[surrogate_name] = fn
        return fn
    return decorator


class SurrogateFactory:
    @staticmethod
    def create(name,**kwargs):
        if name not in _SURROGATES:
            raise ValueError(f"Surrogate {name} not found")
        return _SURROGATES[name](**kwargs)

    @staticmethod
    def serialize(surrogate_instance):
        """
        Serializza un'istanza di surrogato.
        """
        surrogate_type = type(surrogate_instance).__name__
        surrogate_name = next(
            (name for name, cls in _SURROGATES.items() if isinstance(surrogate_instance, cls)),
            None
        )
        if surrogate_name is None:
            raise ValueError(f"Surrogate {surrogate_type} non registrato.")
        return {
            'type': surrogate_name,
            'params': surrogate_instance.__dict__  # Salva lo stato interno del surrogato
        }

    @staticmethod
    def deserialize(data):
        """
        Ricostruisce un'istanza di surrogato dai dati serializzati.
        """
        surrogate_name = data['type']
        if surrogate_name not in _SURROGATES:
            raise ValueError(f"Surrogate {surrogate_name} non registrato.")
        surrogate_cls = _SURROGATES[surrogate_name]
        instance = surrogate_cls.__new__(surrogate_cls)  # Crea l'istanza senza chiamare __init__
        instance.__dict__.update(data['params'])  # Ripristina lo stato interno
        return instance