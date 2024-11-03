import numpy as np
import yaml
import os
from pathlib import Path
from .schedules import DiffusionScheduler

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        self[k] = AttrDict(v)
                    else:
                        self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    self[k] = AttrDict(v)
                else:
                    self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(AttrDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

def expr_constructor(loader, node):
    """Constructor for handling !expr tags in YAML"""
    value = loader.construct_scalar(node)
    # Create a local namespace with numpy
    namespace = {'np': np}
    return eval(value, namespace)

# Create a custom YAML loader class
class ExprLoader(yaml.SafeLoader):
    pass

# Register the custom constructor with our loader
ExprLoader.add_constructor('!expr', expr_constructor)

def load_config(name):
    """Load config from YAML file"""
    config_dir = Path(__file__).parent / 'config'
    config_path = config_dir / f'{name}.yaml'
    
    with open(config_path) as f:
        config = yaml.load(f, Loader=ExprLoader)
    
    # Convert to recursive AttrDict
    config = AttrDict(config)
    
    # Generate schedules based on config
    blur_schedule, noise_schedule = DiffusionScheduler.generate_schedules(
        config,
        config.get('max_step', 200)  # Default to 200 steps if not specified
    )
    
    # Add schedules to config
    config.blur_schedule = blur_schedule
    config.noise_schedule = noise_schedule
    
    return config

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