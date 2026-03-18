import os
import sys
import torch
import scipy.io as sio
import numpy as np
import cv2
import time

# Add local modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models

# To avoid importing utility.__init__ which imports dataset leading to torchnet error:
import importlib.util
spec = importlib.util.spec_from_file_location('indexes', 'utility/indexes.py')
indexes = importlib.util.module_from_spec(spec)
sys.modules['utility.indexes'] = indexes
spec.loader.exec_module(indexes)
MSIQA = indexes.MSIQA

import argparse

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Run QRNN3D inference')
    parser.add_argument('--input_path', type=str, default='dataset/JasperRidge/Case1/data.mat', help='Path to the input data.mat file')
    parser.add_argument('--output_dir', type=str, default='result/JasperRidge/Case1', help='Directory to save the restored results')
    parser.add_argument('--model_type', type=str, default='complex', choices=['gauss', 'complex', 'paviaft'], help='Type of model to use')
    parser.add_argument('--norm', type=str, default='clipped', choices=['minmax', 'clipped', 'raw'], help='Normalization method: minmax (scale 0-1), clipped (clamp 0-1), or raw (none)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 1. Load model
    print(f'Loading model ({args.model_type})...')
    model = models.__dict__['qrnn3d']()
    
    # Resolve checkpoint path based on model type
    checkpoint_map = {
        'gauss': 'checkpoints/qrnn3d/gauss/model_epoch_50_118454.pth',
        'complex': 'checkpoints/qrnn3d/complex/model_epoch_100_159904.pth',
        'paviaft': 'checkpoints/qrnn3d/paviaft/model_epoch_150_160454.pth'
    }
    checkpoint_path = checkpoint_map[args.model_type]
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)
    model.eval()

    # 2. Load dataset
    print(f'Loading dataset from {args.input_path}...')
    if not os.path.exists(args.input_path):
        print(f"Error: Dataset file not found at {args.input_path}")
        return
        
    mat = sio.loadmat(args.input_path)
    
    noisy_hsi = mat['input'][:]  # (H, W, C)
    gt_hsi = mat['gt'][:]
    print('Input shape (H,W,C):', noisy_hsi.shape)

    # Convert to (C, H, W)
    noisy_hsi_chw = noisy_hsi.transpose((2, 0, 1))
    gt_hsi_chw = gt_hsi.transpose((2, 0, 1))
    
    # Construct tensor (B, 1, C, H, W) where B=1
    input_tensor = torch.from_numpy(noisy_hsi_chw[None, None, ...]).float().to(device)
    gt_tensor = torch.from_numpy(gt_hsi_chw[None, None, ...]).float().to(device)
    
    # Apply normalization ONLY to input_tensor, GT remains untouched for evaluation
    print(f'Applying {args.norm} normalization to input...')
    if args.norm == 'minmax':
        # min-max scaling to 0-1
        t_min, t_max = input_tensor.min(), input_tensor.max()
        if t_max > t_min:
            input_tensor = (input_tensor - t_min) / (t_max - t_min)
    elif args.norm == 'clipped':
        # 0以下を0、1以上を1
        input_tensor = torch.clamp(input_tensor, 0.0, 1.0)
    elif args.norm == 'raw':
        pass
    
    # 3. Test
    print('Running inference...')
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 4. Calculate metrics
    print('Calculating metrics...')
    # MSIQA expects BxCxHxW or Bx1xCxHxW, make sure it matches MSIQA expected format
    # utility/indexes calculates metrics based on prediction and target tensors.
    psnr, ssim, sam = MSIQA(output_tensor, gt_tensor)
    print(f'Results - MPSNR: {psnr:.4f}, MSSIM: {ssim:.4f}, SAM: {sam:.4f}')

    # Convert back to (H, W, C)
    # output_tensor is (1, 1, C, H, W) or (1, C, H, W)
    output_np = output_tensor.data[0].cpu().numpy()[0, ...].transpose((1, 2, 0))



    # 7. Save MAT
    print('Saving MAT file...')
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    mat_name = f'{args.model_type}_{args.norm}.mat'
    save_path = os.path.join(save_dir, mat_name)
    
    # MATLABの構造体として保存するために辞書(dict)を使用します
    end_time = time.time()
    running_time = end_time - start_time
    print(f'Total execution time: {running_time:.4f} seconds')

    params_dict = {
        'norma': args.norm,
        'model_type': args.model_type
    }
    
    other_result = {
        'running_time': running_time
    }
    
    # ユーザー指定の変数名で辞書のキーを設定
    sio.savemat(save_path, {
        'HSI_restored': output_np, 
        'hsi_gt': gt_hsi, 
        'HSI_noisy': noisy_hsi,
        'params': params_dict,
        'other_result': other_result,
    })
    print('Saved to:', save_dir)

if __name__ == '__main__':
    main()
