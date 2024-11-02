import pytest
import torch
import numpy as np
#from tfdiff.inference import eval_ssim, cal_SNR_EEG, cal_SNR_MIMO
from inference import eval_ssim, cal_SNR_EEG, cal_SNR_MIMO

class TestMetrics:
    def test_ssim_calculation(self):
        # Arrange
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        height, width = 32, 32
        # Create identical signals for perfect SSIM
        signal = torch.randn(1, 1, height, width, dtype=torch.complex64, device=device)
        
        # Act
        ssim = eval_ssim(signal, signal, height, width, device)
        
        # Assert
        # Use reasonable tolerance for floating point comparison
        assert torch.isclose(ssim, torch.tensor(1.0, device=device), rtol=1e-2, atol=1e-2)

    def test_snr_eeg_calculation(self):
        # Arrange
        # Create perfect signal reconstruction
        truth = torch.randn(10, 100)  # [batch_size, sequence_length]
        predict = truth.clone()
        
        # Act
        snr = cal_SNR_EEG(predict, truth)
        
        # Assert
        assert np.all(np.isinf(snr))  # Perfect reconstruction should give infinite SNR

    def test_snr_mimo_calculation(self):
        # Arrange
        # Create sample MIMO channel data
        batch_size, n_ant, n_subcarriers = 1, 4, 26
        truth = torch.randn(batch_size, n_ant, n_subcarriers, 2)  # Last dim: [real, imag]
        predict = truth.clone()
        
        # Act
        snr = cal_SNR_MIMO(predict, truth)
        
        # Assert
        assert np.isinf(snr)  # Perfect reconstruction should give infinite SNR

    def test_metrics_with_noise(self):
        # Arrange
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        height, width = 32, 32
        signal = torch.randn(1, 1, height, width, dtype=torch.complex64, device=device)
        noisy_signal = signal + 0.1 * torch.randn_like(signal)
        
        # Act
        ssim = eval_ssim(noisy_signal, signal, height, width, device)
        
        # Assert
        assert ssim < 1.0  # SSIM should be less than 1 for noisy signal
        assert ssim > 0.0  # SSIM should be positive

    def test_metrics_edge_cases(self):
        # Test with zero signal
        truth = torch.zeros(10, 100)
        predict = torch.zeros(10, 100)
        
        # This should not raise an error
        snr = cal_SNR_EEG(predict, truth)
        assert not np.any(np.isnan(snr))  # Use np.any() for array comparison

        # Test with very small values
        truth = torch.ones(10, 100) * 1e-10
        predict = truth + 1e-12
        
        snr = cal_SNR_EEG(predict, truth)
        assert not np.isnan(snr)