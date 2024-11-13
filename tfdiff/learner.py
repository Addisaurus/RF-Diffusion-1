import numpy as np
import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from tfdiff.diffusion import SignalDiffusion, GaussianDiffusion
from tfdiff.dataset import _nested_map
from tfdiff.memory_utils import track_memory, clear_memory
from tfdiff.rf_metrics import RFMetrics
import matplotlib.pyplot as plt

class tfdiffLoss(nn.Module):
    def __init__(self, w=0.1):
        super().__init__()
        self.w = w

    def forward(self, target, est, target_noise=None, est_noise=None):
        target_fft = torch.fft.fft(target, dim=1) 
        est_fft = torch.fft(est)
        t_loss = self.complex_mse_loss(target, est)
        f_loss = self.complex_mse_loss(target_fft, est_fft)
        n_loss = self.complex_mse_loss(target_noise, est_noise) if (target_noise and est_noise) else 0.
        return (t_loss + f_loss + self.w * n_loss)

    def complex_mse_loss(self, target, est):
        target = torch.view_as_complex(target)
        est = torch.view_as_complex(est)
        return torch.mean(torch.abs(target-est)**2)
        

class tfdiffLearner:
    def __init__(self, log_dir, model_dir, model, dataset, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.task_id = params.task_id
        self.log_dir = log_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = next(model.parameters()).device if list(model.parameters()) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diffusion = SignalDiffusion(params) if params.signal_diffusion else GaussianDiffusion(params)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=0.5)
        self.params = params
        self.iter = 0
        self.is_master = True
        self.loss_fn = nn.MSELoss()

        # Initialize RF metrics
        self.rf_metrics = RFMetrics()
        
        # For tracking moving averages of metrics
        self.metric_window_size = 100
        self.ssim_history = []
        self.fid_history = []

        # Initialize wandb only on master process
        if self.is_master:
            try:
                # Generate a more descriptive run name
                run_name = f"{type(model).__name__}_b{params.batch_size}_h{params.hidden_dim}_s{params.target_sequence_length}_nh{params.num_heads}_nb{params.num_block}"
                
                # Add timestamp for uniqueness
                from datetime import datetime
                timestamp = datetime.now().strftime("%m%d_%H%M")
                run_name = f"{run_name}_{timestamp}"

                run = wandb.init(
                    project="RF-Diffusion",
                    config={
                        "task_id": params.task_id,
                        "batch_size": params.batch_size,
                        "learning_rate": params.learning_rate,
                        "model_type": type(model).__name__,
                        "max_step": params.max_step,
                        "sample_rate": params.sample_rate,
                        "hidden_dim": params.hidden_dim,
                        "num_heads": params.num_heads,
                        "num_block": params.num_block,
                        "dropout": params.dropout,
                        "signal_diffusion": params.signal_diffusion,
                    },
                    name=run_name
                )

                # Define custom chart layouts using the new API
                wandb.define_metric("train/loss", summary="min")
                wandb.define_metric("metrics/ssim", summary="max")
                wandb.define_metric("metrics/fid", summary="min")
                    
                # Create custom dashboard layout
                wandb.run.log_artifact(
                    wandb.Artifact(
                        "dashboard", type="run_config",
                        metadata={
                            "views": [
                                {
                                    "panel_type": "line",
                                    "panel_title": "Training Metrics",
                                    "fields": ["train/loss", "metrics/ssim", "metrics/fid"]
                                },
                                {
                                    "panel_type": "image",
                                    "panel_title": "Samples",
                                    "fields": ["samples"]
                                }
                            ]
                        }
                    )
                )

                # Try to log diagnostic images
                diagnostic_paths = {
                    'diffusion_schedules': os.path.join('diagnostics', 'diffusion_schedules.png'),
                    'diffusion_progression': os.path.join('diagnostics', 'diffusion_progression.png')
                }

                for name, path in diagnostic_paths.items():
                    if os.path.exists(path):
                        wandb.log({f"diagnostics/{name}": wandb.Image(path)})

                # Watch model with wandb
                wandb.watch(model, log="all", log_freq=100)

            except Exception as e:
                print(f"Warning: Non-critical wandb initialization error: {e}")

    def state_dict(self):
        """Get state dict with proper parameter serialization"""
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
            
        # Convert params to serializable format
        params_dict = {}
        for key, value in vars(self.params).items():
            # Skip private attributes
            if not key.startswith('_'):
                # Handle different types of values
                if isinstance(value, (int, float, str, bool, list, dict)):
                    params_dict[key] = value
                elif isinstance(value, torch.Tensor):
                    params_dict[key] = value.cpu().tolist()
                elif value is None:
                    params_dict[key] = None
                else:
                    # Convert other objects to string representation
                    params_dict[key] = str(value)
                    
        return {
            'iter': self.iter,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                    for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                        for k, v in self.optimizer.state_dict().items()},
            'params': params_dict,  # Use our serializable params dict
        }

    def save_to_checkpoint(self, filename='weights'):
        """Save checkpoint with error handling"""
        save_basename = f'{filename}-{self.iter}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        
        try:
            print("\n=== Saving Checkpoint ===")
            state_dict = self.state_dict()
            
            print("Model state keys:", list(state_dict['model'].keys()))
            print("Optimizer state keys:", list(state_dict['optimizer'].keys()))
            print("Params keys:", list(state_dict['params'].keys()))
            
            # Check for any None values in critical places
            for k, v in state_dict['model'].items():
                if v is None:
                    print(f"Warning: None value found in model state for key: {k}")
                    
            for k, v in state_dict['optimizer'].items():
                if v is None:
                    print(f"Warning: None value found in optimizer state for key: {k}")

            torch.save(state_dict, save_name)
            
            # Log model checkpoint as artifact if using wandb
            if self.is_master and hasattr(self, 'wandb'):
                artifact = wandb.Artifact(
                    name=f"model-checkpoint-{self.iter}", 
                    type="model",
                    description=f"Model checkpoint at iteration {self.iter}"
                )
                artifact.add_file(save_name)
                wandb.log_artifact(artifact)
                    
            # Create symlink on Unix or copy on Windows
            if os.name == 'nt':
                torch.save(state_dict, link_name)
            else:
                if os.path.islink(link_name):
                    os.unlink(link_name)
                os.symlink(save_basename, link_name)
                    
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            if 'state_dict' in locals():
                print("State dict top-level keys:", state_dict.keys())
                print("Model structure:", type(self.model))
                print("Optimizer structure:", type(self.optimizer))
                raise

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.iter = state_dict['iter']

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(
                f'{self.model_dir}/{filename}.pt',
                weights_only=True  # Add this parameter
            )
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self, max_iter=None):
        device = self.device
        # Enable torch backends
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        while True:  # epoch
            for features in tqdm(self.dataset, desc=f'Epoch {self.iter // len(self.dataset)}') if self.is_master else self.dataset:
                if max_iter is not None and self.iter >= max_iter:
                    if self.is_master:
                        wandb.finish()
                    return
                    
                features = _nested_map(features, lambda x: x.to(
                    device) if isinstance(x, torch.Tensor) else x)
                    
                # Track full iteration memory
                # print(f"\n=== Iteration {self.iter} Memory ===")
                loss = self.train_iter(features)
                
                if torch.isnan(loss).any():
                    raise RuntimeError(
                        f'Detected NaN loss at iteration {self.iter}.')
                        
                if self.is_master:
                    if self.iter % 50 == 0:
                        self._write_summary(self.iter, features, loss)
                    if self.iter % (len(self.dataset)) == 0:
                        self.save_to_checkpoint()
                        
                self.iter += 1
                
                # Clear memory between iterations
                clear_memory()
                
            self.lr_scheduler.step()

    def train_iter(self, features):
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            with track_memory():
                self.optimizer.zero_grad()
                data = features['data']  # original data, x_0, [B, N, 2]
                cond = features['cond']  # cond, c, [B, C]
                
                # Clear cache before forward pass
                torch.cuda.empty_cache()
                
                B = data.shape[0]
                
                print(f"\n=== Training Iteration ===")
                print(f"Input data shape: {data.shape}")  # original signal
                print(f"Input condition shape: {cond.shape}")  # condition
                
                # Scale data to match model output range
                data_min, data_max = data.min(), data.max()
                scaled_data = (data - data_min) / (data_max - data_min + 1e-8)
                scaled_data = scaled_data * 8.0 - 4.0  # Scale to [-4, 4]
                print(f"Data ranges - Original:[{data_min:.4f}, {data_max:.4f}], Scaled:[{scaled_data.min():.4f}, {scaled_data.max():.4f}]")
                
                # random diffusion step, [B]
                t = torch.randint(0, self.diffusion.max_step, [B], dtype=torch.int64, device=self.device)
                print(f"Timestep shape: {t.shape}")  # diffusion timestep
                
                # Forward pass with degradation
                degrade_data = self.diffusion.degrade_fn(scaled_data, t, self.task_id)
                predicted = self.model(degrade_data, t, cond)
                
                # Enhanced shape validation and adjustment
                if scaled_data.shape != predicted.shape:
                    if len(predicted.shape) == 4:
                        predicted = predicted.squeeze(2)
                    print(f"Adjusted prediction shape: {predicted.shape}")
                    
                # Verify shapes match with detailed error message
                if scaled_data.shape != predicted.shape:
                    raise ValueError(
                        f"Shape mismatch after adjustment:\n"
                        f"Target shape: {scaled_data.shape}\n"
                        f"Prediction shape: {predicted.shape}\n"
                        f"Raw model output had shape: {predicted.shape if 'predicted' in locals() else 'N/A'}"
                    )
                
                # Compute loss on scaled data
                loss = self.loss_fn(scaled_data, predicted)
                print(f"Loss value: {loss.item()}")
                print(f"Prediction range: [{predicted.min():.4f}, {predicted.max():.4f}]")
                
                # Track memory during backward pass
                loss.backward()
                
                # Gradient clipping
                self.grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.params.max_grad_norm or 1e9)
                
                # After gradient clipping
                print(f"Gradient norm: {self.grad_norm:.4f}")

                # Optimizer step
                self.optimizer.step()
                
                # Clear memory after iteration
                clear_memory()
                
                return loss

    def _compute_metrics(self, real_data, generated_data):
        """Compute metrics with more detailed debugging"""
        with torch.no_grad():
            # real_cpu = real_data.cpu()
            # generated_cpu = generated_data.cpu()
            print("\n=== Computing Metrics ===")
            print(f"Real data shape: {real_data.shape}, dtype: {real_data.dtype}")
            print(f"Generated data shape: {generated_data.shape}, dtype: {generated_data.dtype}")
            
            try:
                # Calculate SSIM directly on GPU
                ssim = self.rf_metrics.complex_ssim(real_data, generated_data)
                print(f"SSIM computed successfully: {ssim.item()}")
                
                # Calculate FID
                fid = self.rf_metrics.rf_fid(real_data, generated_data)
                print(f"FID computed successfully: {fid}")
                
                return ssim.item(), fid
            except Exception as e:
                print(f"Error computing metrics: {str(e)}")
                print(f"Real data range: [{real_data.min()}, {real_data.max()}]")
                print(f"Generated data range: [{generated_data.min()}, {generated_data.max()}]")
                raise

    def _write_summary(self, iter, features, loss):
        if not self.is_master:
            return
            
        # Get a sample batch for metrics
        with torch.no_grad():
            # Generate samples using the current model
            device = next(self.model.parameters()).device
            cond = features['cond'].to(device)
            real_data = features['data'].to(device)
            
            # Generate samples
            noise = torch.randn_like(real_data)
            generated_data = self.model(noise, torch.zeros(noise.shape[0], dtype=torch.long, device=device), cond)
            
            # Compute metrics
            ssim, fid = self._compute_metrics(real_data, generated_data)
            
            # Update metric histories
            self.ssim_history.append(ssim)
            self.fid_history.append(fid)
            
            # Keep only recent values
            if len(self.ssim_history) > self.metric_window_size:
                self.ssim_history.pop(0)
                self.fid_history.pop(0)
            
            # Calculate moving averages
            avg_ssim = sum(self.ssim_history) / len(self.ssim_history)
            avg_fid = sum(self.fid_history) / len(self.fid_history)
        
        metrics = {
            "train/loss": loss.item(),
            "train/grad_norm": self.grad_norm,
            "train/learning_rate": self.optimizer.param_groups[0]['lr'],
            "train/epoch": iter // len(self.dataset),
            "metrics/ssim": ssim,
            "metrics/fid": fid,
            "metrics/ssim_moving_avg": avg_ssim,
            "metrics/fid_moving_avg": avg_fid,
            "system/gpu_utilization": torch.cuda.utilization() if torch.cuda.is_available() else 0,
            "system/gpu_memory_allocated": torch.cuda.memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0,
            "system/gpu_memory_reserved": torch.cuda.memory_reserved(self.device) / 1024**3 if torch.cuda.is_available() else 0,
        }
        
        # Log metrics to wandb
        wandb.log(metrics, step=iter)
        
        # Also log sample visualizations periodically
        if iter % 500 == 0:  # Adjust frequency as needed
            try:
                # Create figure with subplots
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                
                # Plot real and generated samples in time domain
                axes[0,0].plot(real_data[0,:,0].cpu().numpy())
                axes[0,0].set_title('Real Signal (Time Domain)')
                
                axes[0,1].plot(generated_data[0,:,0].cpu().numpy())
                axes[0,1].set_title('Generated Signal (Time Domain)')
                
                # Plot real and generated samples in frequency domain
                real_fft = np.abs(np.fft.fft(real_data[0,:,0].cpu().numpy()))
                gen_fft = np.abs(np.fft.fft(generated_data[0,:,0].cpu().numpy()))
                
                axes[1,0].plot(real_fft)
                axes[1,0].set_title('Real Signal (Frequency Domain)')
                
                axes[1,1].plot(gen_fft)
                axes[1,1].set_title('Generated Signal (Frequency Domain)')
                
                plt.tight_layout()
                
                # Log figure to wandb
                wandb.log({"samples": wandb.Image(fig)}, step=iter)
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Failed to log sample visualizations: {e}")