_SERVERS = {}

def register_server(server_type):
    def decorator(fn):
        _SERVERS[server_type] = fn
        return fn
    return decorator

class ServerFactory:
    @staticmethod
    def create(server_type, **kwargs):
        if server_type not in _SERVERS:
            raise ValueError(f"Unknown server type: {server_type}")
        return _SERVERS[server_type](**kwargs)