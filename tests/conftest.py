import os
# Set OpenMP environment variable before any other imports
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import pytest
import torch
import numpy as np
from pathlib import Path
from tfdiff.params import AttrDict
from tfdiff.conditioning import FieldConfig

@pytest.fixture(scope="session")
def device():
    """Fixture providing device to use for tests"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def sample_config_dir(tmp_path):
    """Create a temporary directory with sample config files"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir

@pytest.fixture
def basic_config():
    """Basic configuration dictionary"""
    return AttrDict({
        'task_id': 4,
        'batch_size': 32,
        'learning_rate': 1.0e-3,
        'max_step': 200,
        'sample_rate': 32768,
        'input_dim': 1,
        'extra_dim': [1],
        'hidden_dim': 256,
        'num_heads': 8,
        'dropout': 0.1,
        'mlp_ratio': 4.0,
        'num_block': 6,
        'embed_dim': 256,
        'learn_tfdiff': False,
        'blur_schedule': np.linspace(0.001, 0.1, 200).tolist(),
        'noise_schedule': np.linspace(1e-4, 0.05, 200).tolist()
    })

@pytest.fixture
def sample_conditioning_config():
    """Sample conditioning configuration"""
    return {
        'enabled_fields': ['mod_type', 'snr', 'symbol_period'],
        'field_configs': {
            'mod_type': FieldConfig(
                type="categorical",
                values=['bpsk', 'qpsk', '8psk', 'dqpsk'],
                required=True
            ),
            'snr': FieldConfig(
                type="continuous",
                normalize=True,
                min_value=-10,
                max_value=30,
                required=False
            ),
            'symbol_period': FieldConfig(
                type="continuous",
                normalize=True,
                min_value=1.0,
                max_value=10.0,
                required=False
            )
        }
    }

@pytest.fixture
def sample_rf_data(device):
    """Generate sample RF data for testing"""
    batch_size = 2
    sample_rate = 1024
    input_dim = 1
    
    # Generate complex-valued sample data
    real = torch.randn(batch_size, sample_rate, input_dim, device=device)
    imag = torch.randn(batch_size, sample_rate, input_dim, device=device)
    data = torch.stack([real, imag], dim=-1)
    
    return data

@pytest.fixture
def mock_model_inputs(device, sample_rf_data, basic_config):
    """Generate mock inputs for model testing"""
    batch_size = sample_rf_data.shape[0]
    return {
        'data': sample_rf_data,
        'timesteps': torch.zeros(batch_size, dtype=torch.long, device=device),
        'cond': torch.randn(batch_size, basic_config.input_dim, device=device)
    }

@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for all tests"""
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("tfdiff.test")

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create temporary directory structure for data testing"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    metadata_file = tmp_path / "metadata.txt"
    metadata_file.touch()
    return data_dir, metadata_file