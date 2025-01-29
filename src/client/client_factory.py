_CLIENTS = {}

def register_client(client_type):
    def decorator(fn):
        _CLIENTS[client_type] = fn
        return fn
    return decorator

class ClientFactory:
    """
    ClientFactory is a factory class for creating client instances.

    Methods:
        create(client_type: str, remote: bool = False, num_gpus: int = 0, **kwargs) -> object:
            Creates and returns an instance of the specified client type.
            
            Parameters:
                client_type (str): The type of client to create.
                remote (bool, optional): If True, creates a remote client instance. Defaults to False.
                num_gpus (int, optional): The number of GPUs to allocate for the client. Defaults to 0.
                **kwargs: Additional keyword arguments to pass to the client constructor.
            
            Returns:
                object: An instance of the specified client type.
            
            Raises:
                ValueError: If the specified client type is not recognized.
    """
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