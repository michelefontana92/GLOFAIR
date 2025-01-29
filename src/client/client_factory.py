_CLIENTS = {}

def register_client(client_type):
    def decorator(fn):
        _CLIENTS[client_type] = fn
        return fn
    return decorator

class ClientFactory:
    @staticmethod
    def create(client_type, remote=False,num_gpus=0,**kwargs):
        if client_type not in _CLIENTS:
            raise ValueError(f"Unknown client type: {client_type}")
        if remote:
            if num_gpus > 0:
                return _CLIENTS[client_type].options(num_cpus=1,num_gpus=num_gpus).remote(**kwargs)
            else:
                return _CLIENTS[client_type].options(num_cpus=1).remote(**kwargs)
        return _CLIENTS[client_type](**kwargs)