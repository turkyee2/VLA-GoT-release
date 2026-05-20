import torch
from torch import nn
# from tqdm import trange # Not used in the provided snippet
import piqa
import lpips
import numpy as np
import scipy.linalg
from typing import Tuple, Dict, List, Optional
import scipy
from torch.cuda.amp import custom_fwd
import cv2
import glob
import os
from tqdm import tqdm
import re # For regex matching

# --- Keep Evaluator and FeatureStats classes as they are ---
class Evaluator(nn.Module):
    def __init__(self, i3d_path='path/pretrained_models/i3d/i3d_torchscript.pt', detector_kwargs=None, max_batchsize=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.lpips = lpips.LPIPS(net='vgg').cuda().eval() # Ensure lpips is on cuda and eval mode
        self.psnr = piqa.PSNR(epsilon=1e-08, value_range=1.0, reduction='none')
        self.ssim = piqa.SSIM(window_size=11, sigma=1.5, n_channels=3, reduction='none').cuda()

        self.i3d_model = torch.jit.load(i3d_path).cuda().eval()
        self.max_batchsize = max_batchsize # Keep max_batchsize if needed for LPIPS batching

        # Default detector_kwargs if not provided
        self.detector_kwargs = detector_kwargs if detector_kwargs is not None else dict(rescale=True, resize=True, return_features=True)

    def compute_fvd(self, real_feature, gen_feature):
        if real_feature.num_items == 0 or gen_feature.num_items == 0:
            return float('nan')  # Return NaN if no data

        mu_real, sigma_real = real_feature.get_mean_cov()
        mu_gen, sigma_gen = gen_feature.get_mean_cov()

        # Add small epsilon to diagonal for numerical stability
        eps = 1e-6
        sigma_real += np.eye(sigma_real.shape[0]) * eps
        sigma_gen += np.eye(sigma_gen.shape[0]) * eps

        try:
            m = np.square(mu_gen - mu_real).sum()
            # Corrected: Compute sqrtm of sigma_real @ sigma_gen, not sigma_gen @ sigma_real
            sqrt_prod, _ = scipy.linalg.sqrtm(np.dot(sigma_real, sigma_gen), disp=False)
            # print(sqrt_prod)

            # Check for imaginary components (shouldn't happen with proper conditioning)
            if np.iscomplexobj(sqrt_prod):
                print("Warning: Complex number encountered in FVD sqrtm. Using real part.")
                sqrt_prod = np.real(sqrt_prod)

            fid = np.real(m + np.trace(sigma_real + sigma_gen - 2 * sqrt_prod))
            return float(fid)
        except Exception as e:
            print(f"Error during FVD calculation: {e}")
            return float('nan')
    

    def get_i3d_features(self, video_tensor):
        """ Extracts I3D features for a batch of videos. """
        # Input video_tensor shape: (B, T, C, H, W), range [0, 1]
        # I3D expects (B, C, T, H, W), range [0, 255]
        if video_tensor is None or video_tensor.numel() == 0:
             return None
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4).contiguous() * 255. # B, C, T, H, W
        with torch.no_grad():
             features = self.i3d_model(video_tensor, **self.detector_kwargs)
        return features.cpu() # Return features on CPU

    # custom_fwd: turn off mixed precision to avoid numerical instability during evaluation
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, video_1, video_2):
        # video_1: ground-truth (B, T, C, H, W) range [0, 1]
        # video_2: reconstruction or prediction (B, T, C, H, W) range [0, 1]
        # Assumes videos are already truncated to the same length T

        # Clamp values to be safe
        video_1 = video_1.clamp(0.0, 1.0)
        video_2 = video_2.clamp(0.0, 1.0)

        if video_1 is None or video_2 is None or video_1.numel() == 0 or video_2.numel() == 0:
            return torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan'))

        B, T, C, H, W = video_1.shape
        video_1_flat = video_1.reshape(B * T, C, H, W)
        video_2_flat = video_2.reshape(B * T, C, H, W)

        with torch.no_grad(): # Ensure no gradients are computed
            mse = self.mse(video_1_flat, video_2_flat).mean([1, 2, 3]).reshape(B, T).mean() # Mean over all frames and batch
            psnr = self.psnr(video_1_flat, video_2_flat).reshape(B, T).mean()
            ssim = self.ssim(video_1_flat, video_2_flat).reshape(B, T).mean()

            # LPIPS expects input range [-1, 1]
            video_1_lpips = video_1_flat * 2 - 1
            video_2_lpips = video_2_flat * 2 - 1

            # Handle potential batch size issues for LPIPS if max_batchsize is set
            # Note: The original batch_forward isn't defined, assuming direct call or simple loop if needed
            if self.max_batchsize is not None and video_1_lpips.shape[0] > self.max_batchsize:
                lpips_val_list = []
                for i in range(0, video_1_lpips.shape[0], self.max_batchsize):
                    batch_1 = video_1_lpips[i:i+self.max_batchsize]
                    batch_2 = video_2_lpips[i:i+self.max_batchsize]
                    lpips_val_list.append(self.lpips(batch_1, batch_2).mean((1, 2, 3))) # Mean over spatial dims
                lpips_val = torch.cat(lpips_val_list).mean() # Mean over all frames and batch
            else:
                 # lpips expects NCHW, input should be B*T, C, H, W
                 lpips_val = self.lpips(video_1_lpips, video_2_lpips).mean() # Already averages over batch and spatial

        return mse, psnr, ssim, lpips_val


class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            # Allow appending features with slightly different num_features?
            # Or enforce strict equality? For I3D it should be constant.
             if num_features != self.num_features:
                 print(f"Warning: Feature dimension mismatch. Expected {self.num_features}, got {num_features}")
                 # Handle mismatch strategy - e.g., skip, resize, error?
                 # For now, let's just warn and proceed if possible.
                 # If resizing is needed, it's complex. Let's assume they match.
                 assert num_features == self.num_features, "Feature dimensions must match"

        else:
            self.num_features = num_features
            if self.capture_all:
                 self.all_features = []
            if self.capture_mean_cov:
                 # Use float64 for accumulators to avoid precision issues
                 self.raw_mean = np.zeros([num_features], dtype=np.float64)
                 self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        # If x is None (e.g., from failed feature extraction), skip
        if x is None:
            return

        # x expected shape (N, D) - N samples, D features
        if x.ndim == 1: # Handle case where only one feature vector is passed
            x = x[np.newaxis, :]
        elif x.ndim != 2:
             raise ValueError(f"Input features must be 2D (N, D), got shape {x.shape}")

        # Ensure numpy array and float32
        x = np.asarray(x, dtype=np.float32)

        if self.num_features is None:
            self.set_num_features(x.shape[1])
        elif x.shape[1] != self.num_features:
            print(f"Warning: Feature dimension mismatch during append. Expected {self.num_features}, got {x.shape[1]}. Skipping append.")
            # Or handle resizing/padding if appropriate, but likely indicates an upstream issue.
            return


        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        if x.shape[0] == 0: # If slicing resulted in empty array
            return

        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            # Ensure x64.T @ x64 results in the correct shape matrix
            self.raw_cov += x64.T @ x64


    def append_torch(self, x_torch: Optional[torch.Tensor]):
        if x_torch is not None and x_torch.numel() > 0:
             # Ensure tensor is on CPU and detached before converting to numpy
             self.append(x_torch.detach().cpu().numpy())

    def get_all(self):
        assert self.capture_all
        if not self.all_features:
             return np.array([], dtype=np.float32).reshape(0, self.num_features if self.num_features is not None else 0)
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        if self.num_items == 0:
             print("Warning: Attempting to get mean/cov with 0 items.")
             # Return zero arrays or raise error? Let's return zeros.
             return np.zeros([self.num_features]), np.zeros([self.num_features, self.num_features])

        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        # Correct covariance calculation: E[X^T X] - E[X]^T E[X]
        cov = cov - np.outer(mean, mean)
        return mean, cov

# --- Modified mp4_to_tensor ---
def mp4_to_tensor(video_path, target_size=None, max_frames=None) -> Optional[torch.Tensor]:
    """
    Reads an MP4 file into a PyTorch Tensor.

    Args:
        video_path (str): Path to the MP4 file.
        target_size (tuple): Target frame size (H, W). Resizes if not None.
        max_frames (int): Maximum number of frames to load. Loads all if None.

    Returns:
        torch.Tensor or None: Video tensor (T, C, H, W), range [0, 1], on CUDA.
                               Returns None if the video cannot be opened or has no frames.
    """
    if not os.path.exists(video_path):
        # print(f"Warning: Video file not found: {video_path}")
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video file: {video_path}")
            return None

        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if target_size is not None:
                # cv2.resize expects (W, H)
                frame = cv2.resize(frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

            frames.append(frame)
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break

        cap.release()

        if not frames:
            print(f"Warning: No frames read from video: {video_path}")
            return None

        # Stack, convert to tensor, permute, normalize
        video_array = np.stack(frames, axis=0) # T, H, W, C
        video_tensor = torch.from_numpy(video_array).cuda().permute(0, 3, 1, 2).float() / 255.0 # T, C, H, W

        # print(f"Loaded video {os.path.basename(video_path)}: {video_tensor.shape}")
        return video_tensor

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None

# --- Helper function to extract common suffix ---
def get_common_suffix(filename):
    """Extracts the common identifying part of the filename."""
    match = re.search(r'(--episode=.*)', filename)
    if match:
        return match.group(1)
    else:
        print(f"Warning: Could not extract common suffix from {filename}")
        return None # Or raise error

# --- Main Evaluation Logic ---
def evaluate_folders(folder_path_1, folder_path_2, i3d_path, target_size=(256, 256)):
    print(f"Evaluating folder 1: {folder_path_1}")
    print(f"Evaluating folder 2: {folder_path_2}")
    print(f"Target size: {target_size}")

    evaluator = Evaluator(i3d_path=i3d_path)
    generation_types = ['gt_recons', 'inf'] # Types of generated videos to compare against GT

    # --- 1. Find video files and map suffixes ---
    file_maps = {folder_path_1: {}, folder_path_2: {}}
    all_suffixes = {folder_path_1: set(), folder_path_2: set()}

    for folder_path in [folder_path_1, folder_path_2]:
        print(f"Scanning folder: {folder_path}")
        for file_path in glob.glob(os.path.join(folder_path, "*.mp4")):
            filename = os.path.basename(file_path)
            suffix = get_common_suffix(filename)
            if suffix:
                file_maps[folder_path][suffix] = file_path
                # Store suffixes based on the 'gt' type to find common base videos
                if '--success=gt--' in suffix:
                     all_suffixes[folder_path].add(suffix)
        print(f"Found {len(file_maps[folder_path])} videos, {len(all_suffixes[folder_path])} 'gt' suffixes.")


    # --- 2. Find common GT suffixes across both folders ---
    common_gt_suffixes = sorted(list(all_suffixes[folder_path_1].intersection(all_suffixes[folder_path_2])))
    print(f"Found {len(common_gt_suffixes)} common 'gt' video suffixes between folders.")

    if not common_gt_suffixes:
        print("Error: No common ground truth videos found between the two folders. Cannot proceed.")
        return

    # --- 3. Initialize Stats and Metric Storage ---
    feature_stats = {
        folder: {
            'gt': FeatureStats(capture_mean_cov=True),
            **{gen_type: FeatureStats(capture_mean_cov=True) for gen_type in generation_types}
        }
        for folder in [folder_path_1, folder_path_2]
    }

    metrics = {
        folder: {
            gen_type: {'mse': [], 'psnr': [], 'ssim': [], 'lpips': []}
            for gen_type in generation_types
        }
        for folder in [folder_path_1, folder_path_2]
    }

    # --- 4. Iterate through common suffixes, load, truncate, evaluate ---
    processed_count = 0
    skipped_count = 0
    for gt_suffix in tqdm(common_gt_suffixes, desc="Processing Videos"):
        # print(f"\nProcessing suffix: {gt_suffix}")
        # Find corresponding files for all types in both folders
        video_paths = {}
        video_tensors = {}
        all_files_found = True
        min_len = float('inf')

        # --- Load all related videos first to find min length ---
        all_related_suffixes = [gt_suffix] + [gt_suffix.replace('--success=gt--', f'--success={gen_type}--') for gen_type in generation_types]

        temp_tensors_for_len = []
        for folder in [folder_path_1, folder_path_2]:
            for suffix in all_related_suffixes:
                 path = file_maps[folder].get(suffix)
                 if path:
                     video_paths[(folder, suffix)] = path
                     tensor = mp4_to_tensor(path, target_size)
                     if tensor is not None and tensor.shape[0] > 0: # Check if tensor is valid and has frames
                         video_tensors[(folder, suffix)] = tensor
                         temp_tensors_for_len.append(tensor)
                     else:
                         # print(f"Warning: Failed to load or empty video for {path}. Skipping this suffix group.")
                         all_files_found = False
                         break # Stop processing this suffix group if one video fails
                 else:
                     # print(f"Warning: Missing file for suffix {suffix} in folder {folder}. Skipping this suffix group.")
                     all_files_found = False
                     break # Stop processing this suffix group if one file is missing
            if not all_files_found:
                 break

        if not all_files_found:
            # print(f"Skipping suffix {gt_suffix} due to missing/invalid files.")
            skipped_count +=1
            # Clear potentially loaded tensors for this group
            del temp_tensors_for_len
            del video_tensors
            del video_paths
            continue # Move to the next gt_suffix

        # Find minimum length across all loaded tensors for this group
        if not temp_tensors_for_len:
            # print(f"Warning: No valid tensors loaded for suffix {gt_suffix}. Skipping.")
            skipped_count += 1
            continue
        
        print([t.shape[0] for t in temp_tensors_for_len])
        min_len = min(t.shape[0] for t in temp_tensors_for_len)
        print(f"Minimum length for suffix group {gt_suffix}: {min_len}")

        if min_len == 0 or min_len <= 21:
             print(f"Warning: Minimum length is 0 for suffix {gt_suffix}. Skipping.")
             skipped_count += 1
             continue

        # --- Truncate and Process ---
        processed_count += 1
        for folder in [folder_path_1, folder_path_2]:
            # Get truncated GT tensor (Batch dimension B=1)
            # gt_tensor_full = video_tensors.get((folder, gt_suffix))
            gt_tensor_full = video_tensors.get((folder_path_1, gt_suffix))
            if gt_tensor_full is None: continue # Should not happen due to checks above, but safety first
            gt_tensor = gt_tensor_full[:min_len, ...].unsqueeze(0) # Add batch dim: 1, T, C, H, W

            # Extract features for GT
            gt_features = evaluator.get_i3d_features(gt_tensor)
            feature_stats[folder]['gt'].append_torch(gt_features)

            # Process generated types
            for gen_type in generation_types:
                gen_suffix = gt_suffix.replace('--success=gt--', f'--success={gen_type}--')
                gen_tensor_full = video_tensors.get((folder, gen_suffix))
                if gen_tensor_full is None: continue # Safety check

                gen_tensor = gen_tensor_full[:min_len, ...].unsqueeze(0) # 1, T, C, H, W

                # Extract features for generated video
                gen_features = evaluator.get_i3d_features(gen_tensor)
                if gen_type != 'gt':
                    feature_stats[folder][gen_type].append_torch(gen_features)

                # Calculate per-video metrics (MSE, PSNR, SSIM, LPIPS)
                mse, psnr, ssim, lpips_val = evaluator(gt_tensor, gen_tensor)

                # Store metrics if they are valid numbers
                if not torch.isnan(mse): metrics[folder][gen_type]['mse'].append(mse.item())
                if not torch.isnan(psnr): metrics[folder][gen_type]['psnr'].append(psnr.item())
                if not torch.isnan(ssim): metrics[folder][gen_type]['ssim'].append(ssim.item() * 100) # Scale SSIM
                if not torch.isnan(lpips_val): metrics[folder][gen_type]['lpips'].append(lpips_val.item() * 100) # Scale LPIPS

        # Clean up tensors for this suffix group to save memory
        del video_tensors
        del video_paths
        del temp_tensors_for_len


    print(f"\nProcessed {processed_count} common video groups.")
    print(f"Skipped {skipped_count} groups due to missing/invalid files or zero length.")

    # --- 5. Calculate and Print Results ---
    results = {}
    for folder_id, folder_path in enumerate([folder_path_1, folder_path_2]):
        folder_name = f"Folder {folder_id+1} ({os.path.basename(folder_path)})"
        print(f"\n--- Results for {folder_name} ---")
        results[folder_name] = {}

        # Calculate FVD
        print("FVD Scores:")
        results[folder_name]['FVD'] = {}
        real_feats = feature_stats[folder_path]['gt']
        for gen_type in generation_types:
            gen_feats = feature_stats[folder_path][gen_type]
            fvd = evaluator.compute_fvd(real_feats, gen_feats)
            results[folder_name]['FVD'][gen_type] = fvd
            print(f"  FVD (gt vs {gen_type}): {fvd:.4f}  (based on {real_feats.num_items} GT, {gen_feats.num_items} Gen samples)")


        # Calculate Average Metrics
        print("\nAverage Metrics:")
        results[folder_name]['Metrics'] = {}
        for gen_type in generation_types:
            print(f"  Metrics for '{gen_type}':")
            results[folder_name]['Metrics'][gen_type] = {}
            for metric_name in ['mse', 'psnr', 'ssim', 'lpips']:
                values = metrics[folder_path][gen_type][metric_name]
                if values:
                    avg_val = sum(values) / len(values)
                else:
                    avg_val = float('nan') # No valid values computed
                results[folder_name]['Metrics'][gen_type][metric_name] = avg_val
                print(f"    Avg {metric_name.upper()}: {avg_val:.4f} (from {len(values)} videos)")

    return results


def main():
    # 1. 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description="Evaluate and compare video generation models from two folders."
    )

    # 2. 添加命令行参数
    parser.add_argument(
        '--i3d_model_path', 
        type=str, 
        default='path/pretrained_models/i3d/i3d_torchscript.pt', # 默认值
        help='Path to the pretrained I3D model. UPDATE THIS DEFAULT PATH or provide via command line.'
    )
    parser.add_argument(
        '--folder_world_model', 
        type=str, 
        required=True, # 设置为必填参数
        help='Path to the output folder of the first model (world model).'
    )
    parser.add_argument(
        '--folder_action_world_model', 
        type=str, 
        required=True, # 设置为必填参数
        help='Path to the output folder of the second model (action world model).'
    )

    # 3. 解析参数
    args = parser.parse_args()

    # 从解析后的参数中获取路径
    i3d_model_path = args.i3d_model_path
    folder_world_model = args.folder_world_model
    folder_action_world_model = args.folder_action_world_model

    # 定义图像尺寸
    img_size = (256, 256)

    # 检查路径是否存在
    if not os.path.exists(i3d_model_path):
        print(f"Error: I3D model not found at {i3d_model_path}")
    elif not os.path.isdir(folder_world_model):
        print(f"Error: World model folder not found at {folder_world_model}")
    elif not os.path.isdir(folder_action_world_model):
        print(f"Error: Action world model folder not found at {folder_action_world_model}")
    else:
        # 运行评估
        print("Starting evaluation...")
        evaluation_results = evaluate_folders(
            folder_world_model, 
            folder_action_world_model, 
            i3d_model_path, 
            target_size=img_size
        )
        print("Evaluation finished.")
        print("Results:", evaluation_results)


if __name__ == "__main__":
    main()

     