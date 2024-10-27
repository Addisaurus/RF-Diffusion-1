import os
import torch
from torch.cuda import device_count
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel
from argparse import ArgumentParser

from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.mimo_model import tfdiff_mimo
from tfdiff.eeg_model import tfdiff_eeg
from tfdiff.fmcw_model import tfdiff_fmcw
from tfdiff.modrec_model import tfdiff_ModRec
from tfdiff.learner import tfdiffLearner
from tfdiff.dataset import from_path
from tfdiff.params import load_config, override_from_args

def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]

def _train_impl(replica_id, model, dataset, params):
    opt = torch.optim.AdamW(model.parameters(), lr=params.training.learning_rate)
    learner = tfdiffLearner(params.log_dir, params.model_dir, model, dataset, opt, params)
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint()
    learner.train(max_iter=params.training.max_iter)

def train(params):
    dataset = from_path(params)
    
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
        if params.training.batch_size % replica_count != 0:
            raise ValueError(
                f'Batch size {params.training.batch_size} is not evenly divisible by # GPUs {replica_count}.')
        params.training.batch_size = params.training.batch_size // replica_count
        port = _get_free_port()
        spawn(train_distributed, args=(replica_count, port, params), nprocs=replica_count, join=True)
    else:
        train(params)

if __name__ == '__main__':
    parser = ArgumentParser(description='Train (or resume training) a tfdiff model')
    add_training_arguments(parser)
    main(parser.parse_args())