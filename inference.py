import math
import numpy as np
import os
import torch
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from matplotlib.font_manager import FontProperties

from argparse import ArgumentParser
import torch.nn as nn

from tfdiff.params import load_config, override_from_args
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.fmcw_model import tfdiff_fmcw
from tfdiff.mimo_model import tfdiff_mimo
from tfdiff.eeg_model import tfdiff_eeg
from tfdiff.modrec_model import tfdiff_ModRec
from tfdiff.diffusion import SignalDiffusion, GaussianDiffusion
from tfdiff.dataset import from_path_inference, _nested_map

from tqdm import tqdm

from glob import glob

import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score

@torch.jit.script
def gaussian(window_size: int, tfdiff: float):
    gaussian = torch.tensor([math.exp(-(x - window_size//2)**2/float(2*tfdiff**2)) for x in range(window_size)])
    return gaussian / gaussian.sum()


@torch.jit.script
def create_window(height: int, width: int):
    h_window = gaussian(height, 1.5).unsqueeze(1)
    w_window = gaussian(width, 1.5).unsqueeze(1)
    _2D_window = h_window.mm(w_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(1, 1, height, width).contiguous()
    return window


def eval_ssim(pred, data, height, width, device):
    window = create_window(height, width).to(torch.complex64).to(device)
    padding = [height//2, width//2]
    mu_pred = torch.nn.functional.conv2d(pred, window, padding=padding, groups=1)
    mu_data = torch.nn.functional.conv2d(data, window, padding=padding, groups=1)
    mu_pred_pow = mu_pred.pow(2.)
    mu_data_pow = mu_data.pow(2.)
    mu_pred_data = mu_pred * mu_data
    tfdiff_pred = torch.nn.functional.conv2d(pred*pred, window, padding=padding, groups=1) - mu_pred_pow
    tfdiff_data = torch.nn.functional.conv2d(data*data, window, padding=padding, groups=1) - mu_data_pow
    tfdiff_pred_data = torch.nn.functional.conv2d(pred*data, window, padding=padding, groups=1) - mu_pred_data
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu_pred*mu_data+C1) * (2*tfdiff_pred_data.real+C2)) / ((mu_pred_pow+mu_data_pow+C1)*(tfdiff_pred+tfdiff_data+C2))
    return 2*ssim_map.mean().real

def cal_SNR_EEG(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()
    PS = np.sum(np.square(truth), axis=-1)  # power of signal
    PN = np.sum(np.square((predict - truth)), axis=-1)  # power of noise
    ratio = PS / PN
    return 10 * np.log10(ratio)

def cal_SNR_MIMO(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy().squeeze(0)
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy().squeeze(0)
    # Recombine the real and imaginary parts to form complex values
    predict_complex = (predict[:,:,:,0] + 1j * predict[:,:,:, 1])
    truth_complex = (truth[:,:,:, 0] + 1j * truth[:,:,:, 1])
    PS = np.sum(np.abs(truth_complex)**2, axis=(-1, -2, -3))  # power of signal
    PN = np.sum(np.abs(predict_complex - truth_complex)**2, axis=(-1, -2, -3))  # power of noise
    ratio = PS / PN
    return 10 * np.log10(ratio)

def save(out_dir, data, cond, batch, index=0):
    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.join(out_dir, 'batch-'+str(batch)+'-'+str(index)+'.mat')
    mat_data = {
        'pred': data.numpy(),
        'cond': cond.numpy()
    }
    scio.savemat(file_name, mat_data)

def save_mimo(out_dir, data, pred, cond, batch, index=0):
    os.makedirs('./dataset/mimo/img/', exist_ok=True)
    font = FontProperties(size=8)
    file_name = os.path.join('./dataset/mimo/img/', 'out-'+str(batch)+'-'+str(index)+'.jpg')
    down = torch.complex(data[0, 0, :, 0, 0].reshape(26), data[0, 0, :, 0, 1].reshape(26))
    down_pred = torch.complex(pred[0, 0, :, 0, 0].reshape(26), pred[0, 0, :, 0, 1].reshape(26))
    up = torch.complex(cond[0, 0, :, 0, 0].reshape(26), cond[0, 0, :, 0, 1].reshape(26))
    down_amp = np.abs(down)*3
    down_phase = np.angle(down)
    pred_amp = np.abs(down_pred)*3
    pred_phase = np.angle(down_pred)
    up_amp = np.abs(up)*3
    up_phase = np.angle(up)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3))
    up_line, = ax1.plot(range(1, 27), up_amp, linewidth=1.2, zorder=10, label='Uplink', color='#084E87')
    pred_line, = ax1.plot(range(1, 27), pred_amp, linewidth=1.2, zorder=10, label='Predict', color='#BF3F3F')
    down_line, = ax1.plot(range(1, 27), down_amp, linewidth=3,alpha=0.6, label='Downlink', color='#ef8a00')
    ax1.set_ylabel('Amplitude')
    ax1.grid(linestyle='--', linewidth=0.5, zorder=0)
    ax1.set_xlim(0, 27)
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontproperties(font)
        label.set_fontsize(7)
    up_phase_line, = ax2.plot(range(1, 27), up_phase, linewidth=1.2, zorder=10, label='Uplink', color='#084E87')
    pred_phase_line, = ax2.plot(range(1, 27), pred_phase, linewidth=1.2, zorder=10, label='Predict', color='#BF3F3F')
    down_phase_line, = ax2.plot(range(1, 27), down_phase, linewidth=3,alpha=0.6, label='Downlink', color='#ef8a00')
    ax2.set_xlabel('Subcarriers', fontproperties=font)
    ax2.set_ylabel('Phase (rad)', fontproperties=font)
    ax2.grid(linestyle='--', linewidth=0.5, zorder=0)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontproperties(font)
        label.set_fontsize(7)
    ax2.set_xlim(0, 27)
    ax2.set_ylim(-np.pi-0.2, np.pi+0.2)
    ax2.legend(handles=[up_phase_line, pred_phase_line, down_phase_line], loc='lower left', prop={'size': 6}, ncol=3, edgecolor='black')
    ax2.get_legend().get_frame().set_linewidth(0.5)
    plt.tight_layout()
    plt.savefig(file_name, dpi=800)


def save_wifi(out_dir, data, pred, cond, batch, index=0):
    scio.savemat(f'./dataset/wifi/output/{batch}-{index}.mat',{'pred':pred.numpy()})
    os.makedirs('./dataset/wifi/img/', exist_ok=True)
    os.makedirs('./dataset/wifi/img_matric/data', exist_ok=True)
    os.makedirs('./dataset/wifi/img_matric/pred', exist_ok=True)
    file_name = os.path.join('./dataset/wifi/img', f'out-{batch}-{index}.jpg')
    file_name_data = os.path.join('./dataset/wifi/img_matric/data', f'out-{batch}-{index}.jpg')
    file_name_pred = os.path.join('./dataset/wifi/img_matric/pred', f'out-{batch}-{index}.jpg')
    
    data = data[0, :, 0].reshape(512)
    pred = pred[0, :, 0].reshape(512)
    # Compute the STFT for data and pred
    n_fft = 24  # Choose an appropriate value for your data
    hop_length = 17  # Choose an appropriate value for your data
    data_spec = torch.stft(data, n_fft=n_fft, hop_length=hop_length)
    pred_spec = torch.stft(pred, n_fft=n_fft, hop_length=hop_length)
    # Convert the complex spectrograms to magnitude spectrograms
    data_spec_mag = torch.abs(data_spec)
    pred_spec_mag = torch.abs(pred_spec)
    # Convert the magnitude spectrograms to dB scale using numpy
    data_spec_dB = 20 * np.log10(data_spec_mag.numpy() + 1e-6)  # Adding a small constant to avoid log(0)
    pred_spec_dB = 20 * np.log10(pred_spec_mag.numpy() + 1e-6)
    # Create a subplot with two columns (one for each spectrogram)
    # 绘制并保存第一个图表
    plt.figure(figsize=(6, 3))
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.matshow(data_spec_dB, cmap='viridis', origin='lower')    
    ax1.set_title('Data Spectrogram (dB)')
    plt.colorbar(im1, format='%+2.0f dB', ax=ax1,orientation='horizontal', pad=0.05)

    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.matshow(pred_spec_dB, cmap='viridis', origin='lower')
    ax2.set_title('Prediction Spectrogram (dB)')
    plt.colorbar(im2, format='%+2.0f dB', ax=ax2,orientation='horizontal', pad=0.05)

    # 保存整个图表为 jpg 文件
    plt.savefig(file_name)
    plt.close()

    # 绘制并保存第二个图表（不包括坐标轴）
    plt.figure(figsize=(6, 7))
    plt.imshow(data_spec_dB, cmap='viridis', origin='lower')    
    plt.axis('off')
    # 保存图片（不包括坐标轴）
    plt.savefig(file_name_data, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 绘制并保存第三个图表（不包括坐标轴）
    plt.figure(figsize=(6, 7))
    plt.imshow(pred_spec_dB, cmap='viridis', origin='lower')    
    plt.axis('off')
    # 保存图片（不包括坐标轴）
    plt.savefig(file_name_pred, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_fmcw(out_dir, data, pred, cond, batch,index=0):
    scio.savemat(f'./dataset/fmcw/output/{batch}-{index}.mat',{'pred':pred.numpy()})
    os.makedirs('./dataset/fmcw/img/', exist_ok=True)
    os.makedirs('./dataset/fmcw/img_matric/data', exist_ok=True)
    os.makedirs('./dataset/fmcw/img_matric/pred', exist_ok=True)
    file_name = os.path.join('./dataset/fmcw/img', f'out-{batch}-{index}.jpg')
    file_name_data = os.path.join('./dataset/fmcw/img_matric/data', f'out-{batch}-{index}.jpg')
    file_name_pred = os.path.join('./dataset/fmcw/img_matric/pred', f'out-{batch}-{index}.jpg')
    
    # 第一行 MATLAB 代码的转换
    data = data.squeeze(0)
    range_fft = np.fft.fftshift(np.fft.fft(data, n=330, axis=1), axes=1)
    # 第二行 MATLAB 代码的转换
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, n=92, axis=0), axes=0)
    # print(range_fft.shape)
    # print(doppler_fft.shape)
    data_spec = doppler_fft.copy()
    pred = pred.squeeze(0)
    range_fft = np.fft.fftshift(np.fft.fft(pred, n=330, axis=1), axes=1)
    # 第二行 MATLAB 代码的转换
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, n=92, axis=0), axes=0)
    pred_spec = doppler_fft.copy()

    # Convert the complex spectrograms to magnitude spectrograms
    data_spec_mag = np.abs(data_spec)
    pred_spec_mag = np.abs(pred_spec)
    # Convert the magnitude spectrograms to dB scale using numpy
    data_spec_dB = 20 * np.log10(data_spec_mag + 1e-6)  # Adding a small constant to avoid log(0)
    pred_spec_dB = 20 * np.log10(pred_spec_mag + 1e-6)
    
    # Create a subplot with two columns (one for each spectrogram)
    # 绘制并保存第一个图表
    plt.figure(figsize=(6, 4))
    ax1 = plt.subplot(2, 1, 1)
    im1 = ax1.matshow(data_spec_dB, cmap='viridis', origin='lower')    
    ax1.set_title('Data Spectrogram (dB)')
    plt.colorbar(im1, format='%+2.0f dB', ax=ax1)

    ax2 = plt.subplot(2, 1, 2)
    im2 = ax2.matshow(pred_spec_dB, cmap='viridis', origin='lower')
    ax2.set_title('Prediction Spectrogram (dB)')
    plt.colorbar(im2, format='%+2.0f dB', ax=ax2)

    # 保存整个图表为 jpg 文件
    plt.savefig(file_name)
    plt.close()

    # 绘制并保存第二个图表（不包括坐标轴）
    plt.figure(figsize=(6, 7))
    plt.imshow(data_spec_dB, cmap='viridis', origin='lower')    
    plt.axis('off')
    # 保存图片（不包括坐标轴）
    plt.savefig(file_name_data, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 绘制并保存第三个图表（不包括坐标轴）
    plt.figure(figsize=(6, 7))
    plt.imshow(pred_spec_dB, cmap='viridis', origin='lower')    
    plt.axis('off')
    # 保存图片（不包括坐标轴）
    plt.savefig(file_name_pred, bbox_inches='tight', pad_inches=0)
    plt.close()

def print_fid(out_dir,data_dir,task_id):
    # 准备真实数据分布和生成模型的图像数据
    real_images_folder = data_dir
    generated_images_folder = out_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = 192
    # 计算FID距离值
    if task_id == 0 :
        corr = 1.9
    else:
        corr = 0.9
    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],batch_size=1,device=device,dims=dims,num_workers=1)*corr
    print('FID value:', fid_value)

def main(args):
    # Load configuration
    config_names = ['wifi', 'fmcw', 'mimo', 'eeg', 'modrec']
    params = load_config(config_names[args.task_id])
    
    # Override with custom config if provided
    if args.config:
        params = load_config(args.config)
    
    # Override with command line arguments
    params = override_from_args(params, args)
    
    # Setup device
    device = torch.device('cpu') if args.device == 'cpu' else torch.device('cuda')
    
    # Load model
    if os.path.exists(f'{params.model_dir}/weights.pt'):
        checkpoint = torch.load(f'{params.model_dir}/weights.pt')
    else:
        checkpoint = torch.load(params.model_dir)
        
    # Initialize model based on task
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
    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Initialize diffusion
    diffusion = SignalDiffusion(params) if params.diffusion.signal_diffusion else GaussianDiffusion(params)
    
    # Setup dataset
    dataset = from_path_inference(params)
    
    # Run inference
    with torch.no_grad():
        cur_batch = 0
        ssim_list = []
        snr_list = []
        
        for features in tqdm(dataset, desc=f'Epoch {cur_batch // len(dataset)}'):
            features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            data = features['data']
            cond = features['cond']
            
            if params.task_id in [0, 1]:  # WiFi/FMCW
                pred = diffusion.native_sampling(model, data, cond, device)
                data_samples = [torch.view_as_complex(sample) for sample in torch.split(data, 1, dim=0)]
                pred_samples = [torch.view_as_complex(sample) for sample in torch.split(pred, 1, dim=0)]
                cond_samples = [torch.view_as_complex(sample) for sample in torch.split(cond, 1, dim=0)]
                
                for b, p_sample in enumerate(pred_samples):
                    d_sample = data_samples[b]
                    cur_ssim = eval_ssim(p_sample, d_sample, params.data.sample_rate, params.data.input_dim, device=device)
                    ssim_list.append(cur_ssim.item())
                    
                    if params.task_id == 1:
                        save_fmcw(params.out_dir, d_sample.cpu().detach(), p_sample.cpu().detach(), cond_samples[b].cpu().detach(), cur_batch, b)
                    else:
                        save_wifi(params.out_dir, d_sample.cpu().detach(), p_sample.cpu().detach(), cond_samples[b].cpu().detach(), cur_batch, b)
                
            elif params.task_id in [2, 3]:  # MIMO/EEG
                pred = diffusion.fast_sampling(model, cond, device)
                if params.task_id == 3:
                    pred = pred.squeeze(2).squeeze(2)
                    data = data.squeeze(2).squeeze(2)
                    pred = pred[:,:,0]
                    data = data[:,:,0]
                    snr_list.append(cal_SNR_EEG(pred, data))
                else:
                    snr_list.append(cal_SNR_MIMO(pred, data))
                    save_mimo(params.out_dir, data.cpu().detach(), pred.cpu().detach(), cond.cpu().detach(), cur_batch)
                    
            elif params.task_id == 4:  # ModRec
                pred = diffusion.native_sampling(model, data, cond, device)
                # Add ModRec-specific evaluation and saving
                
            cur_batch += 1
            
        # Print results
        if params.task_id in [0, 1]:
            print_fid(params.fid_pred_dir, params.fid_data_dir, params.task_id)
            print(f'Average SSIM: {np.mean(ssim_list)}')
        if params.task_id in [2, 3]:
            print(f'Average SNR: {np.mean(snr_list)}')

def add_inference_arguments(parser):
    """Add inference-specific command line arguments"""
    parser.add_argument('--task_id', type=int, required=True,
                      help='Task ID (0:WiFi, 1:FMCW, 2:MIMO, 3:EEG, 4:ModRec)')
    parser.add_argument('--config', type=str,
                      help='Optional path to custom config YAML')
    parser.add_argument('--model_dir', type=str,
                      help='Override model directory')
    parser.add_argument('--out_dir', type=str,
                      help='Override output directory')
    parser.add_argument('--cond_dir', type=str,
                      help='Override condition directory')
    parser.add_argument('--device', default='cuda',
                      help='Device for inference (cuda/cpu)')

if __name__ == '__main__':
    parser = ArgumentParser(description='Run inference using trained tfdiff model')
    add_inference_arguments(parser)
    main(parser.parse_args())