import pytest
import yaml
import numpy as np
from tfdiff.params import load_config, AttrDict, ExprLoader, override_from_args
from argparse import Namespace

class TestConfigLoading:
    def test_basic_config_loading(self, sample_config_dir):
        # Arrange
        config_content = """
        task_id: 4
        batch_size: 32
        learning_rate: 1.0e-3
        """
        config_file = sample_config_dir / "modrec.yaml"
        config_file.write_text(config_content)
        
        # Act
        config = load_config("modrec")
        
        # Assert
        assert isinstance(config, AttrDict)
        assert config.task_id == 4
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3

    def test_numpy_expression_parsing(self, sample_config_dir):
        # Arrange
        config_content = """
        noise_schedule: !expr np.linspace(1e-4, 0.05, 200).tolist()
        blur_schedule: !expr np.linspace(0.001, 0.1, 200).tolist()
        """
        config_file = sample_config_dir / "test.yaml"
        config_file.write_text(config_content)
        
        # Act
        config = yaml.load(config_file.read_text(), Loader=ExprLoader)
        
        # Assert
        assert len(config['noise_schedule']) == 200
        assert len(config['blur_schedule']) == 200
        assert config['noise_schedule'][0] == 1e-4
        assert config['blur_schedule'][-1] == 0.1

    def test_config_override(self, basic_config):
        # Arrange
        args = Namespace(
            batch_size=64,
            learning_rate=2e-3,
            unused_arg="test"
        )
        
        # Act
        config = override_from_args(basic_config, args)
        
        # Assert
        assert config.batch_size == 64
        assert config.learning_rate == 2e-3
        assert not hasattr(config, 'unused_arg')

    def test_invalid_config(self, sample_config_dir):
        # Arrange
        config_content = """
        task_id: invalid
        batch_size: not_a_number
        """
        config_file = sample_config_dir / "invalid.yaml"
        config_file.write_text(config_content)
        
        # Act & Assert
        with pytest.raises(Exception):
            load_config("invalid")