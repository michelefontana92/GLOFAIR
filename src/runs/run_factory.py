from .base_run import BaseRun
_RUNS ={}

def register_run(run):
    def decorator(cls):
        if run in _RUNS:
            raise ValueError(f"Cannot register duplicate run ({run})")
        if not issubclass(cls, BaseRun):
            raise ValueError(f"run ({run}: {cls.__name__}) must extend BaseRun")
        _RUNS[run] = cls
        return cls
    return decorator

class RunFactory:
    @staticmethod
    def create_run(run, **kwargs):
        if run not in _RUNS:
            raise ValueError(f"Unknown run type: {run}")
        return _RUNS[run](**kwargs)
