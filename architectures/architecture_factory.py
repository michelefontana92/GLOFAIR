import torch
_ARCHITECTURES ={}

def register_architecture(architecture):
    def decorator(cls):
        if architecture in _ARCHITECTURES:
            raise ValueError(f"Cannot register duplicate architecture ({architecture})")
        if not issubclass(cls, torch.nn.Module):
            raise ValueError(f"architecture ({architecture}: {cls.__name__}) must extend torch.nn.Module")
        _ARCHITECTURES[architecture] = cls
        return cls
    return decorator

class ArchitectureFactory:
    @staticmethod
    def create_architecture(architecture, **kwargs):
        if architecture not in _ARCHITECTURES:
            raise ValueError(f"Unknown architecture type: {architecture}")
        return _ARCHITECTURES[architecture](**kwargs)
