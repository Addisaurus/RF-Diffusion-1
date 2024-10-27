import numpy as np
import yaml
import os
from pathlib import Path

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self

def yaml_expr_constructor(loader, node):
    """Constructor for handling !expr tags in YAML"""
    value = loader.construct_scalar(node)
    # Create a local namespace with numpy
    namespace = {'np': np}
    return eval(value, namespace)

# Register the custom constructor
yaml.add_constructor('!expr', yaml_expr_constructor)

def load_config(name):
    """Load config from YAML file"""
    config_dir = Path(__file__).parent / 'config'
    config_path = config_dir / f'{name}.yaml'
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return AttrDict(config)

# Load all configurations
params_wifi = load_config('wifi')
params_fmcw = load_config('fmcw')
params_mimo = load_config('mimo')
params_eeg = load_config('eeg')
params_modrec = load_config('modrec')

# Maintain the same interface
all_params = [
    params_wifi,
    params_fmcw,
    params_mimo,
    params_eeg,
    params_modrec
]

# Add CLI override capability
def override_from_args(params, args):
    """Override params with command line arguments"""
    for key, value in vars(args).items():
        if value is not None and hasattr(params, key):
            setattr(params, key, value)
    return params