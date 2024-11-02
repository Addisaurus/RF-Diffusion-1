# test_dataset.py
import pytest
import torch
from tfdiff.dataset import ModRecDataset, WiFiDataset, Collator
from torch.nn import functional as F

class TestDatasets:
    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Create mock .tim files and metadata
        return data_dir, tmp_path / "metadata.txt"

    def test_modrec_dataset_loading(self, sample_data_dir, basic_config):
        """Test ModRecDataset initialization and loading"""
        data_dir, metadata_file = sample_data_dir
        
        # Create sample data
        signal_data = torch.randn(32768).numpy().astype('float32')
        signal_path = data_dir / "signal_001.tim"
        signal_data.tofile(signal_path)
        
        # Create metadata
        metadata = "1 bpsk 1.0 0.0 0.35 1 1 20.0 0.01\n"
        metadata_file.write_text(metadata)
        
        dataset = ModRecDataset([str(data_dir), str(metadata_file)], basic_config)
        assert len(dataset) == 1
        
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert 'data' in sample
        assert 'cond' in sample
        assert sample['data'].shape[-1] == 2  # Complex values

    def test_collator_batch_processing(self, basic_config):
        collator = Collator(basic_config)
        
        # Ensure all tensors are same size after interpolation
        data1 = F.interpolate(torch.randn(1000, 2).unsqueeze(0).permute(0, 2, 1), 
                            basic_config.sample_rate).permute(0, 2, 1).squeeze(0)
        data2 = F.interpolate(torch.randn(1200, 2).unsqueeze(0).permute(0, 2, 1),
                            basic_config.sample_rate).permute(0, 2, 1).squeeze(0)
        
        batch = [
            {'data': data1, 'cond': torch.randn(5, 2)},
            {'data': data2, 'cond': torch.randn(5, 2)}
        ]
        
        result = collator.collate(batch)
        assert result['data'].shape[1] == basic_config.sample_rate

class TestSignalProcessing:
    def test_signal_normalization(self, basic_config):
        """Test signal normalization in collator"""
        collator = Collator(basic_config)
        
        # Create batch with known statistics
        data = torch.randn(1000, 2) * 10 + 5  # Non-standard mean and std
        batch = [{'data': data, 'cond': torch.randn(5, 2)}]
        
        result = collator.collate(batch)
        normalized = result['data'][0]
        
        assert torch.abs(normalized.mean()) < 1e-6  # Should be zero mean
        assert torch.abs(normalized.std() - 1.0) < 1e-6  # Unit variance

    @pytest.mark.parametrize("input_length", [100, 1000, 10000])
    def test_interpolation_consistency(self, basic_config, input_length):
        """Test consistency of signal interpolation"""
        collator = Collator(basic_config)
        
        # Create signals of different lengths
        data = torch.randn(input_length, 2)
        batch = [{'data': data, 'cond': torch.randn(5, 2)}]
        
        result = collator.collate(batch)
        # Check if interpolation happened
        if input_length < basic_config.sample_rate:
            assert result['data'].shape[1] > input_length
        elif input_length > basic_config.sample_rate:
            assert result['data'].shape[1] < input_length