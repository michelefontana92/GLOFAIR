_CLIENTS = {}

def register_client(client_type):
    def decorator(fn):
        _CLIENTS[client_type] = fn
        return fn
    return decorator

class ClientFactory:
    @staticmethod
    def create(client_type, remote=False,**kwargs):
        if client_type not in _CLIENTS:
            raise ValueError(f"Unknown client type: {client_type}")
        if remote:
            return _CLIENTS[client_type].remote(**kwargs)
        return _CLIENTS[client_type](**kwargs)