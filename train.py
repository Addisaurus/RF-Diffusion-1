import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda import device_count
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel
from argparse import ArgumentParser

from tfdiff.other_models.wifi_model import tfdiff_WiFi
from tfdiff.other_models.mimo_model import tfdiff_mimo
from tfdiff.other_models.eeg_model import tfdiff_eeg
from tfdiff.other_models.fmcw_model import tfdiff_fmcw
from tfdiff.modrec_model import tfdiff_ModRec
from tfdiff.learner import tfdiffLearner
from tfdiff.dataset import from_path
from tfdiff.params import load_config, override_from_args
from tfdiff.diffusion import SignalDiffusion, GaussianDiffusion
from tfdiff.debug_utils import configure_logging
import wandb

def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]

def analyze_diffusion_schedules(params):
    """Visualize noise and blur schedules"""
    os.makedirs('diagnostics', exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    # Handle both dictionary and object access
    schedule_config = getattr(params, 'schedule_config', {})
    blur_config = getattr(schedule_config, 'blur_schedule', {}) if schedule_config else {}
    noise_config = getattr(schedule_config, 'noise_schedule', {}) if schedule_config else {}
    
    blur_schedule = getattr(params, 'blur_schedule', [])
    noise_schedule = getattr(params, 'noise_schedule', [])
    
    # Get schedule types safely
    blur_type = blur_config.get('type', 'linear') if isinstance(blur_config, dict) else 'linear'
    noise_type = noise_config.get('type', 'linear') if isinstance(noise_config, dict) else 'linear'
    
    # Plot schedules
    plt.subplot(1, 2, 1)
    plt.plot(blur_schedule, label=f'Blur ({blur_type})')
    plt.title('Blur Schedule')
    plt.xlabel('Diffusion Step')
    plt.ylabel('Blur Level (β)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(noise_schedule, label=f'Noise ({noise_type})')
    plt.title('Noise Schedule')
    plt.xlabel('Diffusion Step')
    plt.ylabel('Noise Level')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join('diagnostics', 'diffusion_schedules.png')
    plt.savefig(save_path)
    
    # Log to wandb if available
    if wandb.run is not None:
        try:
            wandb.log({
                "schedules/analysis": wandb.Image(save_path),
                "schedules/blur_type": blur_type,
                "schedules/noise_type": noise_type,
                "schedules/blur_mean": np.mean(blur_schedule),
                "schedules/noise_std": np.std(noise_schedule),
            })
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")
    
    plt.close()
    
def test_dataset_pickle(params):
    """Test if dataset can be pickled/unpickled"""
    import pickle
    from tfdiff.dataset import ModRecDataset
    print("\n=== Testing Dataset Pickling ===")
    
    try:
        # Create dataset
        print("Creating dataset...")
        dataset = ModRecDataset(params.data_dir, params)
        print("Created dataset successfully")
        
        # Try to pickle
        print("Testing dataset pickle...")
        pickled = pickle.dumps(dataset)
        print(f"Successfully pickled dataset: {len(pickled)} bytes")
        
        # Try to unpickle
        print("Testing dataset unpickle...")
        unpickled = pickle.loads(pickled)
        print("Successfully unpickled dataset")
        
        # Try to load an item
        print("Testing item load...")
        item = unpickled[0]
        print("Successfully loaded item")
        print("\nPickle test passed!")
        
    except Exception as e:
        print(f"\nPickle test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def evaluate_schedules(params, sample_data):
    """Evaluate diffusion schedules on sample data"""
    # Create diagnostics directory if it doesn't exist
    os.makedirs('diagnostics', exist_ok=True)
    
    diffusion = SignalDiffusion(params)
    
    # Simply use the sample_data's device
    device = sample_data.device
    diffusion = diffusion.to(device)
    
    # Test different timesteps
    steps_to_test = [0, params.max_step // 4, params.max_step // 2, 
                     3 * params.max_step // 4, params.max_step - 1]
    
    # print("\n=== Starting Schedule Evaluation ===")
    # print(f"Sample data shape: {sample_data.shape}")
    # print(f"Sample data device: {device}")
    # print(f"Testing steps: {steps_to_test}")
    
    plt.figure(figsize=(15, 10))
    
    for i, t in enumerate(steps_to_test, 1):
        # Apply diffusion
        t_tensor = torch.tensor([t], device=device)
        # print(f"\nProcessing step {t}")
        # print(f"Input tensor shape: {sample_data.shape}")
        
        x_t = diffusion.degrade_fn(sample_data, t_tensor, params.task_id)
        # print(f"Output tensor shape: {x_t.shape}")
        
        # Plot time domain
        plt.subplot(2, len(steps_to_test), i)
        plt.plot(x_t[0, :, 0].cpu().numpy())  # Real part
        plt.title(f't = {t}')
        plt.ylabel('Amplitude' if i == 1 else '')
        
        # Plot frequency domain
        plt.subplot(2, len(steps_to_test), i + len(steps_to_test))
        fft = np.fft.fft(x_t[0, :, 0].cpu().numpy())
        plt.plot(np.abs(fft))
        plt.ylabel('Magnitude' if i == 1 else '')
        plt.xlabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('diagnostics/diffusion_progression.png')
    plt.close()
    
    # Print and save statistical analysis
    # print("\n=== Schedule Analysis ===")
    signal_power = torch.mean(torch.abs(sample_data) ** 2)
    # print(f"Original signal power: {signal_power:.4f}")
    
    analysis_results = []
    for t in steps_to_test:
        t_tensor = torch.tensor([t], device=device)
        x_t = diffusion.degrade_fn(sample_data, t_tensor, params.task_id)
        noise_power = torch.mean(torch.abs(x_t - sample_data) ** 2)
        snr = 10 * torch.log10(signal_power / noise_power)
        result = f"Step {t}: SNR = {snr:.2f} dB"
        # print(result)
        analysis_results.append(result)
    
    # Save analysis results to file
    with open('diagnostics/schedule_analysis.txt', 'w') as f:
        f.write('\n'.join(analysis_results))

def _train_impl(replica_id, model, dataset, params):
    opt = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)
    learner = tfdiffLearner(params.log_dir, params.model_dir, model, dataset, opt, params)
    
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint()

    # Log diagnostic plots to wandb if using it
    if hasattr(learner, 'is_master') and learner.is_master:
        try:
            import wandb
            wandb.log({
                "diffusion_schedules": wandb.Image(os.path.join('diagnostics', 'diffusion_schedules.png')),
                "diffusion_progression": wandb.Image(os.path.join('diagnostics', 'diffusion_progression.png'))
            })
        except Exception as e:
            print(f"Warning: Failed to log diagnostic images to wandb: {e}")
    
    learner.train(max_iter=params.max_iter)

def train(params):
    """Main training function with added diagnostics"""
    print("\n=== GPU Availability Check ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
    #     print(f"CUDA device count: {torch.cuda.device_count()}")
    #     print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")

    # Add the pickle test before creating the dataset
    if params.task_id == 4:  # Only run for ModRec task
        test_dataset_pickle(params)    
        
    # Configure logging based on params
    configure_logging(params)
    # Create diagnostics directory at start
    os.makedirs('diagnostics', exist_ok=True)
    # Plot diffusion schedules before training
    analyze_diffusion_schedules(params)
    
    dataset = from_path(params)
    
    # Get a sample batch for schedule evaluation
    sample_batch = next(iter(dataset))
    sample_data = sample_batch['data'].cuda() if torch.cuda.is_available() else sample_batch['data']
    
    # Evaluate schedules using sample data
    evaluate_schedules(params, sample_data)
    
    # Select model based on task_id
    if params.task_id == 0:
        model = tfdiff_WiFi(params).cuda()
    elif params.task_id == 1:
        model = tfdiff_fmcw(params).cuda()
    elif params.task_id == 2:
        model = tfdiff_mimo(params).cuda()
    elif params.task_id == 3:
        model = tfdiff_eeg(params).cuda()
    elif params.task_id == 4:
        model = tfdiff_ModRec(params).cuda()
    else:
        raise ValueError(f"Unexpected task_id: {params.task_id}")
        
    _train_impl(0, model, dataset, params)

def train_distributed(replica_id, replica_count, port, params):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group(
        'nccl', rank=replica_id, world_size=replica_count)
    
    dataset = from_path(params, is_distributed=True)
    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    
    # Select model based on task_id
    if params.task_id == 0:
        model = tfdiff_WiFi(params).to(device)
    elif params.task_id == 1:
        model = tfdiff_fmcw(params).to(device)
    elif params.task_id == 2:
        model = tfdiff_mimo(params).to(device)
    elif params.task_id == 3:
        model = tfdiff_eeg(params).to(device)
    elif params.task_id == 4:
        model = tfdiff_ModRec(params).to(device)
    else:
        raise ValueError(f"Unexpected task_id: {params.task_id}")
        
    model = DistributedDataParallel(model, device_ids=[replica_id])
    _train_impl(replica_id, model, dataset, params)

def add_training_arguments(parser):
    """Add training-specific command line arguments"""
    parser.add_argument('--task_id', type=int, required=True,
                      help='Task ID (0:WiFi, 1:FMCW, 2:MIMO, 3:EEG, 4:ModRec)')
    parser.add_argument('--config', type=str,
                      help='Optional path to custom config YAML')
    parser.add_argument('--batch_size', type=int,
                      help='Override batch size')
    parser.add_argument('--learning_rate', type=float,
                      help='Override learning rate')
    parser.add_argument('--model_dir', type=str,
                      help='Override model directory')
    parser.add_argument('--data_dir', nargs='+',
                      help='Override data directory')
    parser.add_argument('--log_dir', type=str,
                      help='Override log directory')
    parser.add_argument('--max_iter', type=int,
                      help='Override maximum iterations')

def main(args):
    # Load base configuration for task
    config_names = ['wifi', 'fmcw', 'mimo', 'eeg', 'modrec']
    params = load_config(config_names[args.task_id])
    
    # Override with custom config if provided
    if args.config:
        params = load_config(args.config)
    
    # Override with command line arguments
    params = override_from_args(params, args)
    
    # Distribute across GPUs if available
    replica_count = device_count()
    if replica_count > 1:
        if params.batch_size % replica_count != 0:
            raise ValueError(
                f'Batch size {params.batch_size} is not evenly divisible by # GPUs {replica_count}.')
        params.batch_size = params.batch_size // replica_count
        port = _get_free_port()
        spawn(train_distributed, args=(replica_count, port, params), nprocs=replica_count, join=True)
    else:
        train(params)

if __name__ == '__main__':
    parser = ArgumentParser(description='Train (or resume training) a tfdiff model')
    add_training_arguments(parser)
    main(parser.parse_args())