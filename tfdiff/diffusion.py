import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def print_tensor_info(tensor, name, batch_idx=0):
    """Print detailed tensor information including shape, type, and sample values"""
    # print(f"\n=== {name} ===")
    # print(f"Shape: {tensor.shape}")
    # print(f"Type: {tensor.dtype}")
    # print(f"Device: {tensor.device}")
    # print(f"Min/Max: {tensor.min():.4f} / {tensor.max():.4f}")
    # print(f"Mean/Std: {tensor.mean():.4f} / {tensor.std():.4f}")
    if len(tensor.shape) > 1:
        # print(f"Sample from batch {batch_idx}:")
        sample = tensor[batch_idx] if tensor.shape[0] > batch_idx else tensor[0]
        # print(sample[:5])  # First 5 elements
    # print("=" * 50)

class SignalDiffusion(nn.Module):
    def __init__(self, params):
        super().__init__()
        # print("\n=== Initializing SignalDiffusion ===")
        # print(f"Task ID: {params.task_id}")
        # print(f"Sample Rate: {params.sample_rate}")
        # print(f"Extra Dimensions: {params.extra_dim}")
        # print(f"Max Steps: {params.max_step}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.params = params
        self.task_id = params.task_id
        self.input_dim = self.params.target_sequence_length # input time-series data length, N
        self.extra_dim = self.params.extra_dim # dimension of each data sample, e.g., [S A 2] for complex-valued CSI
        self.max_step = self.params.max_step # maximum diffusion steps
        beta = np.array(self.params.noise_schedule) # \beta, [T]
        self.alpha = torch.tensor((1-beta).astype(np.float32)).to(device) # \alpha_t [T]
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device) # \bar{\alpha_t}, [T]
        self.var_blur = torch.tensor(np.array(self.params.blur_schedule).astype(np.float32)).to(device) # var of blur kernels on the frequency domain for each diffusion step
        self.var_blur_bar = torch.cumsum(self.var_blur, dim=0).to(device) # var of blur kernels on the frequency domain, [T]

        self.var_kernel = (self.input_dim / self.var_blur).unsqueeze(1).to(device) # var of each G_t, [T, 1]
        self.var_kernel_bar = (self.input_dim / self.var_blur_bar).unsqueeze(1).to(device) # var of each \bar{G_t}, [T, 1]
        self.gaussian_kernel = self.get_kernel(self.var_kernel).to(device) # G_t, [T, N]
        self.gaussian_kernel_bar = self.get_kernel(self.var_kernel_bar).to(device) # \bar{G_t}, [T, N]
        # The weight of original information x_0 in degraded data x_t
        self.info_weights = self.gaussian_kernel_bar * torch.sqrt(self.alpha_bar).unsqueeze(-1).to(device) # [T, N]
        # The overall weight of gaussian noise \epsilon in degraded data x_t
        self.noise_weights = self.get_noise_weights().to(device) # [T, N]

        # Debug noise and blur schedules
        # print("\n=== Diffusion Schedules ===")
        # print(f"Noise schedule range: {min(params.noise_schedule):.6f} to {max(params.noise_schedule):.6f}")
        # print(f"Blur schedule range: {min(params.blur_schedule):.6f} to {max(params.blur_schedule):.6f}")

        # print(f"\n=== Weight Shapes ===")
        # print(f"Noise weights shape: {self.noise_weights.shape}")
        # print(f"Info weights shape: {self.info_weights.shape}")
      
    def get_kernel(self, var_kernel):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        samples = torch.arange(0, self.input_dim).to(device) # [N]
        gaussian_kernel = torch.exp(-((samples - self.input_dim // 2)**2) / (2 * var_kernel)) / torch.sqrt(2 * torch.pi * var_kernel) # G_t, [T, N]
        gaussian_kernel = self.input_dim * gaussian_kernel / torch.sum(gaussian_kernel, dim=1, keepdim=True) # Normalized G_t, [T, N]
        return gaussian_kernel

    def get_noise_weights(self):
        noise_weights = []
        for t in range(self.max_step):
            upper_bound = t + 1
            one_minus_alpha_sqrt = torch.sqrt(1 - self.alpha[0:upper_bound]) # \sqrt(1-\bar{\alpha_s}), for s in [1, t], [t]
            rev_one_minus_alpha_sqrt = torch.flipud(one_minus_alpha_sqrt) # \sqrt(1-\bar{\alpha_s}), for s in [t, 1], [t]
            rev_alpha = torch.flipud(self.alpha[0:upper_bound]) # alpha_s, for s in [t, 1], [t]
            rev_alpha_bar_sqrt = torch.sqrt(torch.cumprod(rev_alpha, dim=0) / rev_alpha[-1]) # \sqrt{\bar{\alpha_t} / \bar{\alpha_s}}, for s in [t, 1], [t]
            rev_var_blur = torch.flipud(self.var_blur[:upper_bound]) # [t] 
            rev_var_blur_bar = torch.cumsum(rev_var_blur, dim=0) - rev_var_blur[-1] # [t]
            rev_var_kernel_bar = (self.input_dim / rev_var_blur_bar).unsqueeze(1) # [t, 1]
            rev_kernel_bar = self.get_kernel(rev_var_kernel_bar) # \bar{G_t} / \bar{G_s}, for s in [t, 1], [t, N]
            rev_kernel_bar[0, :] = torch.ones(self.input_dim) 
            noise_weights.append(torch.mv((rev_alpha_bar_sqrt.unsqueeze(-1) * rev_kernel_bar).transpose(0, 1), rev_one_minus_alpha_sqrt)) # [t, N]
        return torch.stack(noise_weights, dim=0) # [T, N] 

    def get_noise_weights_stats(self):
        noise_weights = []
        one_minus_alpha_sqrt = torch.sqrt(1 - self.alpha[0])
        for t in range(self.max_step):
            noise_weights.append((1 - torch.sqrt(self.alpha_bar[t])*self.gaussian_kernel_bar[t, :]) / (1 - torch.sqrt(self.alpha[0]) * self.gaussian_kernel[0, :]))
        return one_minus_alpha_sqrt * torch.stack(noise_weights, dim=0) # [T, N]

    ## Depracated: numerical instable when params.blur_schedule is high, kernel may divided by 0.
    def get_noise_weights_div(self):
        noise_weights = []
        for t in range(self.max_step):
            upper_bound = t + 1
            one_minus_alpha_sqrt = torch.sqrt(1 - self.alpha[:upper_bound]) # \sqrt(1-\bar{\alpha_s}), for s in [1, t], [t]
            ratio_alpha_bar_sqrt = torch.sqrt(self.alpha_bar[t] / self.alpha_bar[:upper_bound]) # \sqrt(\bar{\alpha_t} / \bar{\alpha_s}), for s in [1, t], [t]
            ratio_kernel_bar = self.gaussian_kernel_bar[t, :] / self.gaussian_kernel_bar[:upper_bound, :] # \bar{G_t} / \bar{G_s}, for s in [1, t], [t, N]
            noise_weights.append(torch.mv((ratio_alpha_bar_sqrt.unsqueeze(-1) * ratio_kernel_bar).transpose(0, 1), one_minus_alpha_sqrt)) # [t, N]
        return torch.stack(noise_weights, dim=0) # [T, N]
    
    ## Depracated: numerical instable when params.blur_schedule is high, amplitude of kernel may overflow.
    def get_noise_weights_prod(self):
        noise_weights = []
        for t in range(self.max_step):
            upper_bound = t + 1
            one_minus_alpha_sqrt = torch.sqrt(1 - self.alpha[0:upper_bound]) # \sqrt(1-\bar{\alpha_s}), for s in [1, t], [t]
            rev_one_minus_alpha_sqrt = torch.flipud(one_minus_alpha_sqrt) # \sqrt(1-\bar{\alpha_s}), for s in [t, 1], [t]
            rev_alpha = torch.flipud(self.alpha[0:upper_bound]) # alpha_s, for s in [t, 1], [t]
            rev_alpha_bar_sqrt = torch.sqrt(torch.cumprod(rev_alpha, dim=0) / rev_alpha[-1]) # \sqrt{\bar{\alpha_t} / \bar{\alpha_s}}, for s in [t, 1], [t]
            rev_kernel = torch.flipud(self.gaussian_kernel[:upper_bound, :]) # G_s, for s in [t, 1], [t, N]
            rev_kernel_bar = torch.cumprod(rev_kernel, dim=0) / rev_kernel[-1, :] # \bar{G_t} / \bar{G_s}, for s in [t, 1], [t, N]
            noise_weights.append(torch.mv((rev_alpha_bar_sqrt.unsqueeze(-1) * rev_kernel_bar).transpose(0, 1), rev_one_minus_alpha_sqrt)) # [t, N]
        return torch.stack(noise_weights, dim=0) # [T, N] 

    def degrade_fn(self, x_0, t, task_id):
        """Add noise to the input signal based on the task type"""
        device = x_0.device
        batch_size, seq_len, _ = x_0.shape
        
        # print("\n=== Starting degrade_fn ===")
        # print(f"Input signal shape: {x_0.shape}")
        # print(f"Input signal device: {device}")
        # print(f"Timestep t: {t}")
        # print(f"Timestep device: {t.device}")
        
        # Verify dimensions
        if seq_len != self.input_dim:
            raise ValueError(f"Input sequence length {seq_len} doesn't match target length {self.input_dim}")

        # Convert t to CPU for indexing
        t_cpu = t.cpu()
        
        # Get weights based on task type
        if task_id == 4:  # ModRec
            # Ensure weights match sequence length
            noise_weight = self.noise_weights[t_cpu, :seq_len].unsqueeze(-1).to(device)
            info_weight = self.info_weights[t_cpu, :seq_len].unsqueeze(-1).to(device)
            
            # print(f"Noise weight shape: {noise_weight.shape}")
            # print(f"Info weight shape: {info_weight.shape}")
            # print(f"x_0 shape: {x_0.shape}")
        
        # Generate and apply noise
        noise = torch.randn_like(x_0, dtype=torch.float32, device=device)
        noise = noise_weight * noise  # Will broadcast across batch dimension
        x_t = info_weight * x_0 + noise
        
        # print(f"Output shape: {x_t.shape}")
        return x_t


    def sampling(self, restore_fn, cond, device):
        """Generate new samples"""
        batch_size = cond.shape[0]
        batch_max = (self.max_step-1)*torch.ones(batch_size, dtype=torch.int64)
        
        # Generate initial noise
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        noise = torch.randn(data_dim, dtype=torch.float32, device=device)
        
        # Apply weights based on task type
        if self.task_id in [2, 3]:
            inf_weight = (self.noise_weights[batch_max, :] + self.info_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
        elif self.task_id == 4:
            inf_weight = (self.noise_weights[batch_max, :] + self.info_weights[batch_max, :]).unsqueeze(-1).to(device)
        else:
            inf_weight = (self.noise_weights[batch_max, :] + self.info_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).to(device)
        
        x_s = inf_weight * noise
        
        # Iteratively denoise
        for s in range(self.max_step-1, -1, -1):
            x_0_hat = restore_fn(x_s, s*torch.ones(batch_size, dtype=torch.int64), cond)
            if s > 0:
                x_s = self.degrade_fn(x_0_hat, t=(s-1)*torch.ones(batch_size, dtype=torch.int64), task_id=self.task_id)
                
        return x_0_hat
    
    def robust_sampling(self, restore_fn, cond, device):
        batch_size = cond.shape[0] # B
        batch_max = (self.max_step-1)*torch.ones(batch_size, dtype=torch.int64)
        # Add batch dimension.
        # cond = torch.view_as_real(torch.from_numpy(cond['cond']).to(torch.complex64)).unsqueeze(0)
        # Construct a mini-batch.
        # cond = cond.repeat((batch_size, 1, 1, 1, 1))
        # Generate degraded noise.
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        noise = torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        if self.task_id in [2,3]:
            inf_weight = (self.noise_weights[batch_max, :] + self.info_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        else:
            inf_weight = (self.noise_weights[batch_max, :] + self.info_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        x_s = inf_weight * noise # [B, N, S, A, 2]
        # Restore data from noise.
        for s in range(self.max_step-1, -1, -1): # reverse from t to 0
            x_0_hat = restore_fn(x_s, s*torch.ones(batch_size, dtype=torch.int64), cond) # resotre \hat{x_0} from x_s using trained tfdiff model
            if s > 0:
                # x_{s-1} = x_s - D(\hat{x_0}, s) + D(\hat{x_0}, s-1)
                x_s = x_s - self.degrade_fn(x_0_hat, t=s*torch.ones(batch_size, dtype=torch.int64),task_id = self.task_id) + self.degrade_fn(x_0_hat, t=(s-1)*torch.ones(batch_size, dtype=torch.int64),task_id = self.task_id) # degrade \hat{x_0} to x_{s-1}
        return x_0_hat
        
    def fast_sampling(self, restore_fn, cond, device):
        batch_size = cond.shape[0] # B
        batch_max = (self.max_step-1)*torch.ones(batch_size, dtype=torch.int64)
        # Generate degraded noise.
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        noise = torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        if self.task_id in [2,3]:
            inf_weight = (self.noise_weights[batch_max, :] + self.info_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        else:
            inf_weight = (self.noise_weights[batch_max, :] + self.info_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        x_s = inf_weight * noise # [B, N, S, A, 2]
        # Restore data from noise.
        x_0_hat = restore_fn(x_s, batch_max, cond)
        return x_0_hat
    
    def native_sampling(self, restore_fn, data, cond, device):
        batch_size = cond.shape[0]
        batch_max = (self.max_step-1)*torch.ones(batch_size, dtype=torch.int64)
        # Generate degraded noise.
        x_s = self.degrade_fn(data, batch_max,task_id = self.task_id).to(device)
        # Restore data from noise.
        x_0_hat = restore_fn(x_s, batch_max, cond)
        return x_0_hat

class GaussianDiffusion(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_dim = self.params.sample_rate # input time-series data length, N
        self.extra_dim = self.params.extra_dim # dimension of each data sample, e.g., [S A 2] for complex-valued CSI
        self.max_step = self.params.max_step # maximum diffusion steps
        beta = np.array(self.params.noise_schedule) # \beta, [T]
        alpha = torch.tensor((1-beta).astype(np.float32)) # \alpha_t [T]
        self.alpha_bar = torch.cumprod(alpha, dim=0) # \bar{\alpha_t}, [T]
        # The overall weight of gaussian noise \epsilon in degraded data x_t
        self.noise_weights = torch.sqrt(1 - self.alpha_bar) # \sqrt{1 - \bar{\alpha_t}}, [T]
        self.info_weights = torch.sqrt(self.alpha_bar) # \sqrt{\bar{\alpha_t}}, [T]

    def degrade_fn(self, x_0, t):
        device = x_0.device
        noise_weight = self.noise_weights[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # equivalent gaussian noise weights, [B, 1, 1, 1]
        info_weight = self.info_weights[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # equivalent original info weights, [B, 1, 1, 1] 
        noise = noise_weight * torch.randn_like(x_0, dtype=torch.float32, device=device) # [B, N, S, 2]
        # noise =  noise_weight.unsqueeze(-1).unsqueeze(-1) * torch.randn_like(x_0, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        x_t = info_weight * x_0 + noise # [B, N, S, A, 2]
        return x_t

    def sampling(self, restore_fn, cond, device):
        batch_size = cond.shape[0] # B
        # Generate degraded noise.
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        inf_weight = (self.noise_weights[self.max_step-1] + self.info_weights[self.max_step-1]).to(device) # scalar
        x_s = inf_weight * torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, 2]
        # Restore data from noise.
        for s in range(self.max_step-1, -1, -1): # reverse from t to 0
            x_0_hat = restore_fn(x_s, s*torch.ones(batch_size, dtype=torch.int64), cond) # resotre \hat{x_0} from x_s using trained tfdiff model
            if s > 0:
                # x_{s-1} = D(\hat{x_0}, s-1)
                x_s = self.degrade_fn(x_0_hat, t=(s-1)*torch.ones(batch_size, dtype=torch.int64)) # degrade \hat{x_0} to x_{s-1}
        return x_0_hat
    
    def robust_sampling(self, restore_fn, cond, device):
        batch_size = cond.shape[0] # B
        # Generate degraded noise.
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        inf_weight = (self.noise_weights[self.max_step-1] + self.info_weights[self.max_step-1]).to(device) # scalar
        x_s = inf_weight * torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        # Restore data from noise.
        for s in range(self.max_step-1, -1, -1): # reverse from t to 0
            x_0_hat = restore_fn(x_s, s*torch.ones(batch_size, dtype=torch.int64), cond) # resotre \hat{x_0} from x_s using trained tfdiff model
            if s > 0:
                # x_{s-1} = x_s - D(\hat{x_0}, s) + D(\hat{x_0}, s-1)
                x_s = x_s - self.degrade_fn(x_0_hat, t=[s]) + self.degrade_fn(self, x_0_hat, t=(s-1)*torch.ones(batch_size, dtype=torch.int64)) # degrade \hat{x_0} to x_{s-1}
        return x_0_hat

    def fast_sampling(self, restore_fn, cond, device):
        batch_size = cond.shape[0] # B
        batch_max = (self.max_step-1)*torch.ones(batch_size, dtype=torch.int64)
        # Generate degraded noise.
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        noise = torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        inf_weight = (self.noise_weights[self.max_step-1] + self.info_weights[self.max_step-1]).to(device) # scalar
        x_s = inf_weight * noise # [B, N, S, A, 2]
        # Restore data from noise.
        x_0_hat = restore_fn(x_s, batch_max, cond)
        return x_0_hat
    
    def native_sampling(self, restore_fn, data, cond, device):
        batch_size = cond.shape[0]
        batch_max = (self.max_step-1)*torch.ones(batch_size, dtype=torch.int64)
        # Generate degraded noise.
        x_s = self.degrade_fn(data, batch_max).to(device)
        # Restore data from noise.
        x_0_hat = restore_fn(x_s, batch_max, cond)
        return x_0_hat