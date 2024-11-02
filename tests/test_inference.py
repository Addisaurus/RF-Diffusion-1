import pytest
import torch
import numpy as np
from tfdiff.params import AttrDict
from tfdiff.modrec_model import tfdiff_ModRec
from tfdiff.diffusion import SignalDiffusion
from tfdiff.conditioning import ConditioningManager

class TestInference:
    @pytest.fixture
    def model_setup(self, basic_config, sample_conditioning_config):
        # Combine configs
        config = basic_config
        config.conditioning = AttrDict(sample_conditioning_config)
        
        # Initialize model and diffusion
        model = tfdiff_ModRec(config)
        diffusion = SignalDiffusion(config)
        
        return model, diffusion, config

    def test_model_initialization(self, model_setup):
        # Arrange
        model, _, config = model_setup
        
        # Assert
        assert isinstance(model, tfdiff_ModRec)
        assert model.hidden_dim == config.hidden_dim
        assert model.num_heads == config.num_heads

    def test_forward_pass(self, model_setup, sample_rf_data):
        # Arrange
        model, diffusion, config = model_setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        data = sample_rf_data.to(device)
        
        # Create conditioning
        manager = ConditioningManager(config)
        cond = manager.create_condition_vector({
            'mod_type': 'bpsk',
            'snr': 10.0
        }).to(device)
        
        # Create timesteps
        timesteps = torch.zeros(data.shape[0], dtype=torch.long, device=device)
        
        # Act
        with torch.no_grad():
            output = model(data, timesteps, cond)
        
        # Assert
        assert output.shape == data.shape
        assert not torch.isnan(output).any()

    def test_inference_pipeline(self, model_setup, sample_rf_data):
        # Arrange
        model, diffusion, config = model_setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        data = sample_rf_data.to(device)
        
        manager = ConditioningManager(config)
        cond = manager.create_condition_vector({
            'mod_type': 'bpsk',
            'snr': 10.0
        }).to(device)
        
        # Act
        with torch.no_grad():
            generated = diffusion.native_sampling(model, data, cond, device)
        
        # Assert
        assert generated.shape == data.shape
        assert not torch.isnan(generated).any()