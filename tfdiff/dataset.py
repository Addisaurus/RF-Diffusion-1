import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import scipy.io as scio
from tfdiff.params import AttrDict
from glob import glob
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import struct

# data_key='csi_data',
# gesture_key='gesture',
# location_key='location',
# orient_key='orient',
# room_key='room',
# rx_key='rx',
# user_key='user',


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)

# New dataset class
class ModRecDataset(torch.utils.data.Dataset):
    """Dataset for CSPB.ML.2018R2 radio modulation signals.
    Each sample contains I/Q samples and conditioning information including
    modulation type, symbol period, carrier offset, etc."""
    
    def __init__(self, paths):
        super().__init__()
        # Verify we have both data and metadata paths
        data_path = paths[0]  # Directory containing .tim files
        metadata_file = paths[1]  # signal_record_first_20000.txt
        
        # Load metadata
        self.metadata = pd.read_csv(
            metadata_file, 
            sep='\s+',
            names=[
                'id', 'modulation', 'symbol_period', 'carrier_offset',
                'excess_bw', 'upsample', 'downsample', 'snr', 'n0'
            ]
        )
        
        # Map modulations to indices
        self.modulations = [
            'bpsk', 'qpsk', '8psk', 'dqpsk', 
            '16qam', '64qam', '256qam', 'msk'
        ]
        
        # Get all .tim files
        self.filenames = glob(f'{data_path}/**/signal_*.tim', recursive=True)
        if not self.filenames:
            raise RuntimeError(f"No .tim files found in {data_path}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Get signal file
        filename = self.filenames[idx]
        signal_id = int(filename.split('_')[-1].split('.')[0])
        
        # Get corresponding metadata
        params = self.metadata[self.metadata['id'] == signal_id].iloc[0]
        
        # Read binary signal data
        try:
            with open(filename, 'rb') as fp:
                # Read raw I/Q samples
                rawdata = np.fromfile(fp, dtype=np.float32)
                # Convert to complex
                signal = rawdata[::2] + 1j * rawdata[1::2]
                signal = torch.from_numpy(signal).to(torch.complex64)
        except Exception as e:
            raise IOError(f"Error reading {filename}: {e}")

        # Create conditioning tensor
        cond = torch.tensor([
            self.modulations.index(params.modulation),
            params.symbol_period,
            params.carrier_offset,
            params.excess_bw,
            params.snr
        ], dtype=torch.float32)

        return {
            'data': torch.view_as_real(signal),  # [N, 2]
            'cond': cond  # [5]
        }

class WiFiDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            self.filenames += glob(f'{path}/**/user*.mat', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cur_filename = self.filenames[idx]
        cur_sample = scio.loadmat(cur_filename,verify_compressed_data_integrity=False)
        # cur_data = torch.from_numpy(cur_sample['csi_data']).to(torch.complex64)
        cur_data = torch.from_numpy(cur_sample['feature']).to(torch.complex64)
        cur_cond = torch.from_numpy(cur_sample['cond']).to(torch.complex64)
        return {
            'data': cur_data,
            'cond': cur_cond.squeeze(0)
        }
    

class FMCWDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            self.filenames += glob(f'{path}/**/*.mat', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cur_filename = self.filenames[idx]
        cur_sample = scio.loadmat(cur_filename)
        cur_data = torch.from_numpy(cur_sample['feature']).to(torch.complex64)
        cur_cond = torch.from_numpy(cur_sample['cond'].astype(np.int16)).to(torch.complex64)
        return {
            'data': cur_data,
            'cond': cur_cond.squeeze(0)
        }

class MIMODataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    super().__init__()
    self.filenames = []
    for path in paths:
        self.filenames += glob(f'{path}/**/*.mat', recursive=True)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self,idx):
    dataset = scio.loadmat(self.filenames[idx])
    data = torch.from_numpy(dataset['down_link']).to(torch.complex64)
    cond = torch.from_numpy(dataset['up_link']).to(torch.complex64)
    return {
        'data': torch.view_as_real(data),
        'cond': torch.view_as_real(cond)
    }


class EEGDataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    super().__init__()
    paths = paths[0]
    self.filenames = []
    self.filenames += glob(f'{paths}/*.mat', recursive=True)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self,idx):
    dataset = scio.loadmat(self.filenames[idx])
    data = torch.from_numpy(dataset['clean']).to(torch.complex64)
    cond = torch.from_numpy(dataset['disturb']).to(torch.complex64)
    return {
        'data': data,
        'cond': cond
    }


class Collator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):
        sample_rate = self.params.sample_rate
        task_id = self.params.task_id

        if task_id == 4:  # ModRec task
            for record in minibatch:
                data = record['data']  # [N, 2]
                # Ensure consistent length through downsampling/interpolation
                if len(data) > sample_rate:
                    down_sample = F.interpolate(
                        data.permute(1, 0).unsqueeze(0),
                        sample_rate, 
                        mode='linear'
                    )
                    data = down_sample.squeeze(0).permute(1, 0)
                # Normalize
                norm_data = (data - data.mean()) / data.std()
                record['data'] = norm_data

            # Stack batch
            data = torch.stack([record['data'] for record in minibatch])
            cond = torch.stack([record['cond'] for record in minibatch])
            
            # Convert condition to complex-like format by adding zero imaginary part
            cond_complex = torch.stack([cond, torch.zeros_like(cond)], dim=-1)
            
            return {
                'data': data,  # [B, N, 2] 
                'cond': cond_complex  # [B, 5, 2]
            }
        ## WiFi Case
        elif task_id == 0:
            for record in minibatch:
                # Filter out records that aren't long enough.
                if len(record['data']) < sample_rate:
                    del record['data']
                    del record['cond']
                    continue
                data = torch.view_as_real(record['data']).permute(1, 2, 0)
                down_sample = F.interpolate(data, sample_rate, mode='nearest-exact')
                norm_data = (down_sample - down_sample.mean()) / down_sample.std()
                record['data'] = norm_data.permute(2, 0, 1)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': data,
                'cond': torch.view_as_real(cond),
            }
        ## FMCW Case
        elif task_id == 1:
            for record in minibatch:
                # Filter out records that aren't long enough.
                if len(record['data']) < sample_rate:
                    del record['data']
                    del record['cond']
                    continue
                data = torch.view_as_real(record['data']).permute(1, 2, 0)
                down_sample = F.interpolate(data, sample_rate, mode='nearest-exact')
                norm_data = (down_sample - down_sample.mean()) / down_sample.std()
                record['data'] = norm_data.permute(2, 0, 1)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': data,
                'cond': torch.view_as_real(cond),
            }

        ## MIMO Case
        elif task_id == 2:
            for record in minibatch:
                data = record['data']
                cond = record['cond']
                # print(f'data.shape:{data.shape}')
                norm_data = (data) / cond.std()
                norm_cond = (cond) / cond.std()
                record['data'] = norm_data.reshape(14, 96, 26, 2).transpose(1,2)
                record['cond'] = norm_cond.reshape(14, 96, 26, 2).transpose(1,2)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': data,
                'cond': cond,
            } 

        ## EEG Case
        if task_id == 3:
            for record in minibatch:
                data = record['data']
                cond = record['cond']

                norm_data = data / cond.std()
                norm_cond = cond / cond.std()
                
                record['data'] = norm_data.reshape(512, 1, 1)
                record['cond'] = norm_cond.reshape(512)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': torch.view_as_real(data),
                'cond': torch.view_as_real(cond),
            } 

        else:
            raise ValueError("Unexpected task_id.")


def from_path(params, is_distributed=False):
    data_dir = params.data_dir
    task_id = params.task_id
    if task_id == 0:
        dataset = WiFiDataset(data_dir)
    elif task_id == 1:
        dataset = FMCWDataset(data_dir)
    elif task_id == 2:
        dataset = MIMODataset(data_dir)
    elif task_id == 3:
        dataset = EEGDataset(data_dir)
    elif task_id == 4:  # Add ModRec case
        dataset = ModRecDataset(data_dir)
    else:
        raise ValueError("Unexpected task_id.")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).collate,
        shuffle=not is_distributed,
        num_workers=8,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True)


def from_path_inference(params):
    cond_dir = params.cond_dir
    task_id = params.task_id
    if task_id == 0:
        dataset = WiFiDataset(cond_dir)
    elif task_id == 1:
        dataset = FMCWDataset(cond_dir)
    elif task_id == 2:
        dataset = MIMODataset(cond_dir)
    elif task_id == 3:
        dataset = EEGDataset(cond_dir)
    else:
        raise ValueError("Unexpected task_id.")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.inference_batch_size,
        collate_fn=Collator(params).collate,
        shuffle=False,
        num_workers=os.cpu_count()
        )
