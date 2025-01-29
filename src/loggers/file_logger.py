from .base_logger import BaseLogger
import os
from icecream import ic
from .logger_factory import register_logger

@register_logger("file")
class FileLogger(BaseLogger):
    def __init__(self, **kwargs):
        file_path = kwargs['file_path']
        file_dir = kwargs.get('file_dir', './logs')
        include_context = kwargs.get('include_context', False)
        prefix = kwargs.get('prefix', '')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        self.file_path = os.path.join(file_dir, file_path)
        self.include_context = include_context
        self.prefix = prefix
        self.reset()

    def reset(self):
        with open(self.file_path, 'w') as file:
            file.write('')
    
    def log(self, message):
        def output_fn(message):
            with open(self.file_path, 'a') as file:
                file.write(f'{message}\n')
        
        ic.configureOutput(prefix=self.prefix,
                           includeContext=self.include_context,
                           outputFunction=output_fn)
        
        ic(message)

    def error(self, message):
        self.log(f'[ERROR] {message}')

    def info(self, message):
        self.log(f'[INFO] {message}')

    def debug(self, message):
        self.log(f'[DEBUG] {message}')
    
    def close(self):
        pass