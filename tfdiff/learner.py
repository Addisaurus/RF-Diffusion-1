import numpy as np
import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from tfdiff.diffusion import SignalDiffusion, GaussianDiffusion
from tfdiff.dataset import _nested_map


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

        # Initialize wandb only on master process
        if self.is_master:
            wandb.init(
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
                name=f"task_{params.task_id}_{type(model).__name__}"
            )
            # Log model architecture
            wandb.watch(model, log="all", log_freq=100)

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'iter': self.iter,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()},
            'params': dict(self.params),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.iter = state_dict['iter']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.iter}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        
        # Log model checkpoint as artifact
        if self.is_master:
            artifact = wandb.Artifact(
                name=f"model-checkpoint-{self.iter}", 
                type="model",
                description=f"Model checkpoint at iteration {self.iter}"
            )
            artifact.add_file(save_name)
            wandb.log_artifact(artifact)
            
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self, max_iter=None):
        device = self.device
        while True:  # epoch
            for features in tqdm(self.dataset, desc=f'Epoch {self.iter // len(self.dataset)}') if self.is_master else self.dataset:
                if max_iter is not None and self.iter >= max_iter:
                    if self.is_master:
                        wandb.finish()
                    return
                features = _nested_map(features, lambda x: x.to(
                    device) if isinstance(x, torch.Tensor) else x)
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
            self.lr_scheduler.step()

    def train_iter(self, features):
        self.optimizer.zero_grad()
        data = features['data']  # original data, x_0, [B, N, S*A, 2]
        cond = features['cond']  # cond, c, [B, C]
        B = data.shape[0]
        # random diffusion step, [B]
        t = torch.randint(0, self.diffusion.max_step, [B], dtype=torch.int64, device=self.device)
        degrade_data = self.diffusion.degrade_fn(
            data, t, self.task_id)  # degrade data, x_t, [B, N, S*A, 2]
        predicted = self.model(degrade_data, t, cond)
        if self.task_id==3:
            data = data.reshape(-1,512,1,2)
        loss = self.loss_fn(data, predicted)
        loss.backward()
        self.grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.optimizer.step()
        return loss

    def _write_summary(self, iter, features, loss):
        if not self.is_master:
            return
            
        metrics = {
            "train/loss": loss.item(),
            "train/grad_norm": self.grad_norm,
            "train/learning_rate": self.optimizer.param_groups[0]['lr'],
            "train/epoch": iter // len(self.dataset),
            "system/gpu_utilization": torch.cuda.utilization() if torch.cuda.is_available() else 0,
            "system/gpu_memory_allocated": torch.cuda.memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0,  # Convert to GB
            "system/gpu_memory_reserved": torch.cuda.memory_reserved(self.device) / 1024**3 if torch.cuda.is_available() else 0,
        }
        
        wandb.log(metrics, step=iter)