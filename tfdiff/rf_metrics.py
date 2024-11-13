import torch
import numpy as np
import torch.nn.functional as F

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
    def create_gaussian_window(window_size, sigma=1.5, device='cuda'):
        """Create a Gaussian window for SSIM calculation"""
        gauss = torch.exp(-((torch.arange(window_size, device=device) - (window_size-1)/2)**2)/(2*sigma**2))
        return gauss/gauss.sum()
    
    @staticmethod
    def _compute_complex_ssim(x1, x2, window_size, size_average, val_range):
        """Compute SSIM for complex signals with improved stability"""
        device = x1.device
        window = RFMetrics.create_gaussian_window(window_size, device=device)
        window = window.to(dtype=torch.complex64)  # Make window complex
        
        def complex_conv1d(x, window):
            # Ensure x is complex
            if not torch.is_complex(x):
                raise ValueError(f"Expected complex tensor, got {x.dtype}")
            
            return F.conv1d(
                x.view(-1, 1, x.size(-1)),
                window.view(1, 1, -1),
                padding=window_size//2
            ).squeeze(1)
        
        # Determine value range if not provided
        if val_range is None:
            val_range = max(torch.abs(x1).max(), torch.abs(x2).max())

        # Constants for stability
        C1 = (0.01 * val_range)**2
        C2 = (0.03 * val_range)**2

        # Create initial tensors on correct device
        zero = torch.zeros(1, device=device, dtype=torch.float32)
                
        # Create complex constants directly on the correct device
        C1 = torch.complex(
            torch.full_like(zero, C1),  # Real part filled with C1 value
            zero                        # Imaginary part filled with 0
        )
        C2 = torch.complex(
            torch.full_like(zero, C2),  # Real part filled with C2 value
            zero                        # Imaginary part filled with 0
        )

        # Compute means
        mu1 = complex_conv1d(x1, window)
        mu2 = complex_conv1d(x2, window)
        
        # Compute variances and covariance using complex arithmetic
        mu1_sq = mu1.conj() * mu1
        mu2_sq = mu2.conj() * mu2
        mu1_mu2 = mu1.conj() * mu2
        
        # Compute variances using complex arithmetic
        sigma1_sq = complex_conv1d(x1.conj() * x1, window) - mu1_sq
        sigma2_sq = complex_conv1d(x2.conj() * x2, window) - mu2_sq
        sigma12 = complex_conv1d(x1.conj() * x2, window) - mu1_mu2

        # Compute SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator
        
        # Take real part for final SSIM value 
        #ssim_map = torch.real(ssim_map)
        ssim_map = torch.abs(ssim_map)
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1)

    @staticmethod
    def complex_ssim(x1, x2, window_size=11, size_average=True, val_range=None):
        """
        Calculate SSIM for complex-valued RF signals
        
        Args:
            x1 (torch.Tensor): First complex signal [B, N, 2] (real, imag)
            x2 (torch.Tensor): Second complex signal [B, N, 2] (real, imag)
            window_size (int): Size of the sliding window
            size_average (bool): If True, average SSIM across batch
            val_range (float): Maximum value range. If None, determined from data
            
        Returns:
            torch.Tensor: SSIM value(s)
        """
        if not x1.is_cuda and torch.cuda.is_available():
            print("Warning: Input tensors are on CPU but CUDA is available. Consider keeping tensors on GPU for better performance.")

        print("\n=== Complex SSIM Debug ===")
        print(f"Input x1 shape: {x1.shape}, dtype: {x1.dtype}, device: {x1.device}")
        print(f"Input x2 shape: {x2.shape}, dtype: {x2.dtype}, device: {x2.device}")
        print(f"x1 contiguous: {x1.is_contiguous()}")
        print(f"x1 values first element: {x1[0,0]}")

        # Validate inputs
        RFMetrics.validate_input(x1, "x1")
        RFMetrics.validate_input(x2, "x2")

        # Convert to complex tensors properly
        x1_complex = torch.view_as_complex(x1.contiguous())
        x2_complex = torch.view_as_complex(x2.contiguous())

        print(f"Complex tensor shapes - x1:{x1_complex.shape}, x2:{x2_complex.shape}")
        
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
        
        # Combine metrics
        result = 0.5 * (ssim_time + ssim_freq)
        print(f"SSIM Result - Time:{ssim_time.item()}, Freq:{ssim_freq.item()}, Combined:{result.item()}")
        
        return result

    @staticmethod
    def extract_rf_features(signals_complex):
        """
        Extract comprehensive RF-specific features from complex signals.
        Returns: Tensor of shape [batch_size, feature_dim]
        """
        # Time domain features
        inst_amplitude = torch.abs(signals_complex)  # [batch, seq]
        inst_phase = torch.angle(signals_complex)    # [batch, seq]
        
        # Frequency domain features
        freq_domain = torch.fft.fft(signals_complex, dim=1)
        freq_magnitude = torch.abs(freq_domain)      # [batch, seq]
        freq_phase = torch.angle(freq_domain)        # [batch, seq]
        
        # Compute statistics along sequence dimension
        features = torch.stack([
            inst_amplitude.mean(dim=1),              # Mean amplitude
            inst_amplitude.std(dim=1),               # Std amplitude
            inst_phase.mean(dim=1),                  # Mean phase
            inst_phase.std(dim=1),                   # Std phase
            freq_magnitude.mean(dim=1),              # Mean frequency magnitude
            freq_magnitude.std(dim=1),               # Std frequency magnitude
            torch.cos(freq_phase).mean(dim=1),       # Mean cosine of frequency phase
            torch.sin(freq_phase).mean(dim=1),       # Mean sine of frequency phase
        ], dim=1)  # Shape: [batch_size, 8]
        
        return features
    
    @staticmethod
    def rf_fid(real_data, generated_data, device='cuda'):
        """
        Calculate enhanced FID score for RF signals.
        """
        # Validate inputs
        RFMetrics.validate_input(real_data, "real_data")
        RFMetrics.validate_input(generated_data, "generated_data")
        
        # Convert to complex
        real_complex = torch.view_as_complex(real_data.contiguous())
        gen_complex = torch.view_as_complex(generated_data.contiguous())
        
        # Extract features
        real_features = RFMetrics.extract_rf_features(real_complex)
        gen_features = RFMetrics.extract_rf_features(gen_complex)
        
        # Ensure features have correct shape [batch_size, feature_dim]
        if real_features.dim() == 1:
            real_features = real_features.unsqueeze(0)
        if gen_features.dim() == 1:
            gen_features = gen_features.unsqueeze(0)
            
        print(f"\n=== FID Computation Debug ===")
        print(f"Feature shapes - Real:{real_features.shape}, Gen:{gen_features.shape}")
        
        # Normalize features
        real_features = (real_features - real_features.mean(0)) / (real_features.std(0) + 1e-8)
        gen_features = (gen_features - gen_features.mean(0)) / (gen_features.std(0) + 1e-8)
        
        # Calculate mean and covariance
        mu1, mu2 = real_features.mean(0), gen_features.mean(0)
        sigma1 = torch.cov(real_features.permute(1, 0))  # [feature_dim, batch_size]
        sigma2 = torch.cov(gen_features.permute(1, 0))
        
        print(f"Covariance shapes - Sigma1:{sigma1.shape}, Sigma2:{sigma2.shape}")
        
        # Calculate FID with improved numerical stability
        diff = mu1 - mu2
        
        # Add small offset to diagonal for numerical stability
        offset = torch.eye(sigma1.size(0), device=device) * 1e-6
        sigma1 = sigma1 + offset
        sigma2 = sigma2 + offset
        
        # Compute square root using eigendecomposition
        try:
            # Compute sigma1 @ sigma2
            sigma_product = sigma1.mm(sigma2)
            
            # Compute eigendecomposition
            eigvals, eigvecs = torch.linalg.eigh(sigma_product)
            
            # Ensure eigenvalues are positive and take sqrt
            sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=1e-8))
            
            # Compute matrix square root
            covmean = eigvecs.mm(torch.diag(sqrt_eigvals)).mm(eigvecs.t())
            
            print(f"Covmean shape: {covmean.shape}")
            
            tr_covmean = torch.trace(covmean)
            
            # Compute FID
            fid = torch.real(diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean)
            
            return fid.item()
            
        except Exception as e:
            print(f"Error in FID computation: {str(e)}")
            print(f"Eigenvalues: {eigvals}")
            print(f"Min eigenvalue: {eigvals.min()}")
            print(f"Max eigenvalue: {eigvals.max()}")
            raise

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