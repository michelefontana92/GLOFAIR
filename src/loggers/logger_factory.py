_LOGGERS = {}

def register_logger(logger_type):
    def decorator(fn):
        _LOGGERS[logger_type] = fn
        return fn
    return decorator

class LoggerFactory:
    @staticmethod
    def create_logger(logger_type, **kwargs):
        if logger_type not in _LOGGERS:
            raise ValueError(f"Unknown logger type: {logger_type}")
        return _LOGGERS[logger_type](**kwargs)