# test_error_handling.py
import pytest
import torch
from tfdiff.dataset import ModRecDataset
from tfdiff.conditioning import ConditioningManager
from tfdiff.dataset import Collator

class TestErrorHandling:
    def test_corrupt_signal_file(self, sample_data_dir, basic_config):
        """Test handling of corrupted signal files"""
        data_dir, metadata_file = sample_data_dir
        
        # Create corrupted signal file
        signal_path = data_dir / "signal_001.tim"
        signal_path.write_bytes(b"corrupted data")
        
        metadata_file.write_text("1 bpsk 1.0 0.0 0.35 1 1 20.0 0.01\n")
        
        dataset = ModRecDataset([str(data_dir), str(metadata_file)], basic_config)
        with pytest.raises(IOError):
            _ = dataset[0]

    def test_invalid_condition_combinations(self, full_config):
        """Test handling of invalid conditioning combinations"""
        cm = ConditioningManager(full_config)
        
        with pytest.raises(ValueError):
            cm.create_condition_vector({
                'mod_type': 'bpsk',
                'snr': 'invalid',  # Should be numeric
            })

    def test_memory_efficiency(self, basic_config):
        """Test memory usage during batch processing"""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss
        
        # Create large batch
        large_batch = [
            {'data': torch.randn(10000, 2), 'cond': torch.randn(5, 2)}
            for _ in range(32)
        ]
        
        collator = Collator(basic_config)
        _ = collator.collate(large_batch)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 1e9  # Less than 1GB