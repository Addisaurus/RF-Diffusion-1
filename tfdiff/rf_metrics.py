import torch
import numpy as np

class RFMetrics:
    """Metrics specifically designed for RF signals with improved accuracy and stability"""
    
    @staticmethod
    def validate_input(signals, name="signals"):
        """Validate input tensor format and values"""
        if not isinstance(signals, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if signals.dim() != 3:
            raise ValueError(f"{name} must be 3D tensor [B, N, 2], got shape {signals.shape}")
        if signals.size(-1) != 2:
            raise ValueError(f"{name} last dimension must be 2 (real, imag), got {signals.size(-1)}")
        if not torch.isfinite(signals).all():
            raise ValueError(f"{name} contains non-finite values")

    @staticmethod
    def check_numerical_stability(tensor, name):
        """Check for numerical stability issues"""
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite values detected in {name}")
        if torch.abs(tensor).max() > 1e6:
            raise ValueError(f"Potentially unstable large values detected in {name}")

    @staticmethod
    def complex_ssim(x1, x2, window_size=11, size_average=True, val_range=None):
        """
        Calculate SSIM for complex-valued RF signals with improved accuracy.
        
        Args:
            x1 (torch.Tensor): First complex signal [B, N, 2] (real, imag)
            x2 (torch.Tensor): Second complex signal [B, N, 2] (real, imag)
            window_size (int): Size of the sliding window
            size_average (bool): If True, average SSIM across batch
            val_range (float): Maximum value range. If None, determined from data
            
        Returns:
            torch.Tensor: SSIM value(s)
        """
        # Validate inputs
        RFMetrics.validate_input(x1, "x1")
        RFMetrics.validate_input(x2, "x2")
        
        # Convert to complex numbers
        x1_complex = torch.complex(x1[..., 0], x1[..., 1])
        x2_complex = torch.complex(x2[..., 0], x2[..., 1])
        
        # Calculate in time domain
        ssim_time = RFMetrics._compute_complex_ssim(
            x1_complex, x2_complex, window_size, size_average, val_range
        )
        
        # Calculate in frequency domain
        x1_freq = torch.fft.fft(x1_complex, dim=1)
        x2_freq = torch.fft.fft(x2_complex, dim=1)
        ssim_freq = RFMetrics._compute_complex_ssim(
            x1_freq, x2_freq, window_size, size_average, val_range
        )
        
        # Combine time and frequency domain metrics
        return 0.5 * (ssim_time + ssim_freq)
    
    @staticmethod
    def create_gaussian_window(window_size, sigma=1.5, device='cuda'):
        """Create a Gaussian window for SSIM calculation"""
        gauss = torch.exp(-((torch.arange(window_size, device=device) - (window_size-1)/2)**2)/(2*sigma**2))
        return gauss/gauss.sum()
    
    @staticmethod
    def _compute_complex_ssim(x1, x2, window_size, size_average, val_range):
        """
        Compute SSIM for complex signals with improved stability.
        """
        device = x1.device
        window = RFMetrics.create_gaussian_window(window_size, device=device)
        
        def complex_conv1d(x, window):
            return torch.conv1d(x.view(-1, 1, x.size(-1)), 
                              window.view(1, 1, -1), 
                              padding=window_size//2).squeeze(1)
        
        # Compute means
        mu1 = complex_conv1d(x1, window)
        mu2 = complex_conv1d(x2, window)
        
        # Compute variances and covariance
        sigma1_sq = complex_conv1d(torch.abs(x1 - mu1)**2, window)
        sigma2_sq = complex_conv1d(torch.abs(x2 - mu2)**2, window)
        sigma12 = complex_conv1d((x1 - mu1).conj() * (x2 - mu2), window)
        
        # Determine value range if not provided
        if val_range is None:
            val_range = max(torch.abs(x1).max(), torch.abs(x2).max())
        
        # Constants for stability
        C1 = (0.01 * val_range)**2
        C2 = (0.03 * val_range)**2
        
        # Compute SSIM
        numerator = (2 * torch.abs(mu1 * mu2.conj()) + C1) * (2 * torch.abs(sigma12) + C2)
        denominator = (torch.abs(mu1)**2 + torch.abs(mu2)**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator
        
        RFMetrics.check_numerical_stability(ssim_map, "SSIM map")
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1)
    
    @staticmethod
    def extract_rf_features(signals_complex):
        """
        Extract comprehensive RF-specific features from complex signals.
        """
        # Time domain features
        inst_amplitude = torch.abs(signals_complex)
        inst_phase = torch.angle(signals_complex)
        inst_freq = torch.diff(inst_phase, dim=1)
        
        # Frequency domain features
        freq_domain = torch.fft.fft(signals_complex, dim=1)
        freq_magnitude = torch.abs(freq_domain)
        freq_phase = torch.angle(freq_domain)
        
        # Statistical features
        features = torch.cat([
            torch.mean(inst_amplitude, dim=1),
            torch.std(inst_amplitude, dim=1),
            torch.mean(inst_freq, dim=1),
            torch.std(inst_freq, dim=1),
            torch.mean(freq_magnitude, dim=1),
            torch.std(freq_magnitude, dim=1),
            torch.mean(torch.cos(freq_phase), dim=1),
            torch.std(torch.cos(freq_phase), dim=1)
        ], dim=-1)
        
        return features
    
    @staticmethod
    def rf_fid(real_signals, generated_signals, device='cuda'):
        """
        Calculate enhanced FID score for RF signals.
        """
        # Validate inputs
        RFMetrics.validate_input(real_signals, "real_signals")
        RFMetrics.validate_input(generated_signals, "generated_signals")
        
        # Convert to complex
        real_complex = torch.complex(real_signals[..., 0], real_signals[..., 1])
        gen_complex = torch.complex(generated_signals[..., 0], generated_signals[..., 1])
        
        # Extract features
        real_features = RFMetrics.extract_rf_features(real_complex)
        gen_features = RFMetrics.extract_rf_features(gen_complex)
        
        # Normalize features
        real_features = (real_features - real_features.mean(0)) / (real_features.std(0) + 1e-8)
        gen_features = (gen_features - gen_features.mean(0)) / (gen_features.std(0) + 1e-8)
        
        # Calculate mean and covariance
        mu1, sigma1 = real_features.mean(0), torch.cov(real_features.T)
        mu2, sigma2 = gen_features.mean(0), torch.cov(gen_features.T)
        
        # Calculate FID with improved numerical stability
        diff = mu1 - mu2
        
        offset = torch.eye(sigma1.size(0), device=device) * 1e-6
        sigma1 = sigma1 + offset
        sigma2 = sigma2 + offset
        
        covmean = torch.matrix_power(sigma1.mm(sigma2), 0.5)
        
        tr_covmean = torch.trace(covmean)
        
        fid = torch.real(diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean)
        
        RFMetrics.check_numerical_stability(fid, "FID score")
        
        return fid.item()

    @staticmethod
    def frequency_dependent_phase_coherence(real_complex, gen_complex):
        """
        Calculate frequency-dependent phase coherence between two signals.
        """
        real_fft = torch.fft.fft(real_complex, dim=1)
        gen_fft = torch.fft.fft(gen_complex, dim=1)
        
        phase_diff = torch.angle(gen_fft) - torch.angle(real_fft)
        magnitude_weights = torch.abs(real_fft)
        
        coherence = torch.sum(torch.cos(phase_diff) * magnitude_weights) / torch.sum(magnitude_weights)
        return coherence.real.item()

    @staticmethod
    def evaluate_rf_signals(real_signals, generated_signals, device='cuda'):
        """
        Comprehensive evaluation of RF signal quality with enhanced metrics.
        """
        # Validate inputs
        RFMetrics.validate_input(real_signals, "real_signals")
        RFMetrics.validate_input(generated_signals, "generated_signals")
        
        # Calculate SSIM
        ssim_score = RFMetrics.complex_ssim(
            real_signals, 
            generated_signals,
            size_average=True
        )
        
        # Calculate FID
        fid_score = RFMetrics.rf_fid(
            real_signals,
            generated_signals,
            device
        )
        
        # Convert to complex
        real_complex = torch.complex(real_signals[..., 0], real_signals[..., 1])
        gen_complex = torch.complex(generated_signals[..., 0], generated_signals[..., 1])
        
        # Calculate normalized PSD error
        real_psd = torch.abs(torch.fft.fft(real_complex, dim=1))**2
        gen_psd = torch.abs(torch.fft.fft(gen_complex, dim=1))**2
        
        real_psd_norm = real_psd / (torch.sum(real_psd, dim=1, keepdim=True) + 1e-8)
        gen_psd_norm = gen_psd / (torch.sum(gen_psd, dim=1, keepdim=True) + 1e-8)
        psd_error = torch.mean((real_psd_norm - gen_psd_norm)**2).item()
        
        # Calculate frequency-dependent phase coherence
        phase_coherence = RFMetrics.frequency_dependent_phase_coherence(real_complex, gen_complex)
        
        # Additional RF-specific metrics
        inst_freq_real = torch.diff(torch.angle(real_complex), dim=1)
        inst_freq_gen = torch.diff(torch.angle(gen_complex), dim=1)
        freq_error = torch.mean((inst_freq_real - inst_freq_gen)**2).item()
        
        metrics = {
            'ssim': ssim_score.item(),
            'fid': fid_score,
            'psd_error': psd_error,
            'phase_coherence': phase_coherence,
            'instantaneous_frequency_error': freq_error
        }
        
        return metrics