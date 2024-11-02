import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import pytest
import torch
import numpy as np
from dataclasses import dataclass
from tfdiff.conditioning import ConditioningManager, FieldConfig

@dataclass
class MockConfig:
    """Mock config class for testing"""
    class Conditioning:
        def __init__(self, enabled_fields, field_configs):
            self.enabled_fields = enabled_fields
            self.field_configs = field_configs
    
    def __init__(self, enabled_fields, field_configs):
        self.conditioning = self.Conditioning(enabled_fields, field_configs)

@pytest.fixture
def device():
    """Fixture for device handling"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def basic_config():
    """Basic configuration with only mod_type"""
    return MockConfig(
        enabled_fields=['mod_type'],
        field_configs={
            'mod_type': FieldConfig(
                type="categorical",
                values=['bpsk', 'qpsk', '8psk'],
                required=True
            )
        }
    )

@pytest.fixture
def full_config():
    """Full configuration with all fields"""
    return MockConfig(
        enabled_fields=['mod_type', 'snr', 'carrier_offset'],
        field_configs={
            'mod_type': FieldConfig(
                type="categorical",
                values=['bpsk', 'qpsk', '8psk'],
                required=True
            ),
            'snr': FieldConfig(
                type="continuous",
                normalize=True,
                min_value=-10,
                max_value=30,
                required=False
            ),
            'carrier_offset': FieldConfig(
                type="continuous",
                normalize=True,
                min_value=-500,
                max_value=500,
                required=False
            )
        }
    )

class TestConditioningManager:
    def test_initialization(self, basic_config, full_config):
        """Test basic initialization"""
        cm = ConditioningManager(basic_config)
        assert cm.conditioning_dim == 3  # 3 mod types
        
        cm = ConditioningManager(full_config)
        assert cm.conditioning_dim == 5  # 3 mod types + 2 continuous

    def test_device_handling(self, basic_config, device):
        """Test that conditioning vectors are created on the correct device"""
        cm = ConditioningManager(basic_config)
        cond = cm.create_condition_vector({'mod_type': 'bpsk'})
        assert cond.device == torch.device('cpu')
        
        if torch.cuda.is_available():
            cond = cond.to(device)
            assert cond.is_cuda

    def test_missing_mod_type(self):
        """Test that mod_type is required"""
        bad_config = MockConfig(
            enabled_fields=['snr'],
            field_configs={
                'snr': FieldConfig(type="continuous", required=False)
            }
        )
        with pytest.raises(ValueError):
            ConditioningManager(bad_config)

    def test_categorical_conditioning(self, basic_config):
        """Test categorical field conditioning"""
        cm = ConditioningManager(basic_config)
        
        for i, mod_type in enumerate(['bpsk', 'qpsk', '8psk']):
            cond = cm.create_condition_vector({'mod_type': mod_type})
            assert cond[i].item() == 1.0
            assert torch.sum(cond).item() == 1.0
            assert cond.shape[0] == 3

    def test_continuous_conditioning(self, full_config):
        """Test continuous field conditioning with normalization"""
        cm = ConditioningManager(full_config)
        
        # Test boundary values
        test_cases = [
            # (snr, carrier_offset, expected_snr_norm, expected_offset_norm)
            (-10.0, -500.0, 0.0, 0.0),  # Min values
            (30.0, 500.0, 1.0, 1.0),    # Max values
            (10.0, 0.0, 0.5, 0.5),      # Mid values
        ]
        
        for snr, offset, exp_snr, exp_offset in test_cases:
            cond = cm.create_condition_vector({
                'mod_type': 'bpsk',
                'snr': snr,
                'carrier_offset': offset
            })
            assert pytest.approx(cond[3].item(), 1e-5) == exp_snr
            assert pytest.approx(cond[4].item(), 1e-5) == exp_offset

    def test_missing_optional_field(self, full_config):
        """Test handling of missing optional fields"""
        cm = ConditioningManager(full_config)
        cond = cm.create_condition_vector({'mod_type': 'bpsk'})
        assert cond.shape[0] == 5
        assert cond[3].item() == 0.0  # Default SNR
        assert cond[4].item() == 0.0  # Default carrier offset

    def test_invalid_categorical_value(self, basic_config):
        """Test error on invalid categorical value"""
        cm = ConditioningManager(basic_config)
        with pytest.raises(ValueError, match="Invalid value"):
            cm.create_condition_vector({'mod_type': 'invalid_mod'})

    def test_out_of_range_continuous(self, full_config):
        """Test handling of out-of-range continuous values"""
        cm = ConditioningManager(full_config)
        
        # Test value clamping
        cond = cm.create_condition_vector({
            'mod_type': 'bpsk',
            'snr': -20.0,  # Below min
            'carrier_offset': 1000.0  # Above max
        })
        assert cond[3].item() == 0.0  # Clamped to min
        assert cond[4].item() == 1.0  # Clamped to max

    def test_vector_description(self, full_config):
        """Test conversion back to human-readable form"""
        cm = ConditioningManager(full_config)
        original = {
            'mod_type': 'bpsk',
            'snr': 10.0,
            'carrier_offset': 0.0
        }
        cond = cm.create_condition_vector(original)
        described = cm.describe_vector(cond)
        
        assert described['mod_type'] == original['mod_type']
        assert pytest.approx(described['snr'], 1e-5) == original['snr']
        assert pytest.approx(described['carrier_offset'], 1e-5) == original['carrier_offset']