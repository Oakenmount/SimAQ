import argparse
import os
import omegaconf
from skimage import io
from skimage import measure
from watershed import watershed
from scipy import ndimage as ndi
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser(description="Post-processing script for model outputs.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input file containing model outputs.")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="Directory to save the post-processed outputs.")
    parser.add_argument("--config", type=str, default="", help="Path to the configuration file.")

    args = parser.parse_args()
    return args

def default_config():
    return omegaconf.OmegaConf.create({
        "channels": {
            "membrane": {
                "value": 1,
                "apply_watershed": True,
                "min_radius": 70,  # 70 voxels
                "fill_holes": True,
                "smooth_iterations": 4,
                "blur_sigma": 5.0,
                "contained_by": [],
            },
            "vacuole": {
                "value": 2,
                "apply_watershed": True,
                "min_radius": 3,  # 4 voxels, some can be very small in certain phenotypes with fragmented vacuoles
                "fill_holes": True,
                "smooth_iterations": 3,
                "blur_sigma": 5.0,
                "contained_by": ["membrane"],
            },
            "lipid_droplet": {
                "value": 3,
                "apply_watershed": True,
                "min_radius": 2,  # 4 voxels
                "fill_holes": True,
                "smooth_iterations": 2,
                "blur_sigma": 1.0,
                "contained_by": ["membrane", "vacuole"],
            },
        }
    })

def get_subvolume(volume: np.ndarray):
    """Isolate mask of a subvolume that is the smallest bounding box around non-zero elements.
    Args:
        volume (np.ndarray): Input 3D volume.
    Returns:
        np.ndarray: Subvolume containing only non-zero elements.
        np.ndarray: Coordinates of the subvolume.
    
    """
    non_zero_coords = np.argwhere(volume > 0)
    if non_zero_coords.size == 0:
        return volume, None

    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0) + 1  # +1 to include the last index
    subvolume = volume[tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))]
    
    return subvolume, (min_coords, max_coords)

import torch
import torch.nn.functional as F


def gaussian_kernel_3d(sigma: float, size: int) -> torch.Tensor:
    """Create a 3D Gaussian kernel."""
    coords = torch.arange(size) - size // 2
    grid_z, grid_y, grid_x = torch.meshgrid(coords, coords, coords, indexing='ij')
    kernel = torch.exp(-(grid_x**2 + grid_y**2 + grid_z**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def smooth_instances_gpu(instances: torch.Tensor,
                         sigma: float = 1.0,
                         morph_opening: bool = True,
                         opening_iters: int = 1) -> torch.Tensor:
    """
    Smooth 3D instance labels on the GPU using Gaussian blur + optional opening.

    Args:
        instances (torch.Tensor): 3D tensor (H, W, D) with integer instance labels.
        sigma (float): Gaussian smoothing sigma.
        morph_opening (bool): Whether to apply binary erosion + dilation.
        opening_iters (int): Iterations for morphological opening.

    Returns:
        torch.Tensor: Smoothed 3D label volume (same shape), on CPU.
    """
    device = instances.device
    smoothed = torch.zeros_like(instances)

    labels = torch.unique(instances)
    labels = labels[labels != 0]  # Skip background

    # Create Gaussian kernel
    kernel_size = int(2 * round(2 * sigma) + 1)
    kernel = gaussian_kernel_3d(sigma, kernel_size).to(device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    for label_id in labels:
        mask = (instances == label_id).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        # Optional: morphological opening (approximate via min and max pooling)
        if morph_opening:
            for _ in range(opening_iters):
                mask = -F.max_pool3d(-mask, kernel_size=3, stride=1, padding=1)  # Erosion
            for _ in range(opening_iters):
                mask = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)   # Dilation

        # Apply Gaussian smoothing
        padded = F.pad(mask, [kernel_size // 2] * 6, mode='replicate')
        blurred = F.conv3d(padded, kernel)

        # Threshold back to binary and assign label
        mask_smoothed = (blurred > 0.5).squeeze()
        smoothed[mask_smoothed] = label_id

    return smoothed.cpu()


def process_volume(seg_volume: np.ndarray, config) -> np.ndarray:
    seg_out = np.zeros_like(seg_volume, dtype=np.uint8)
    print(seg_out.shape)
    for chan, chan_cfg in config.channels.items():
        print(f"Processing channel: {chan}")
        # Check values that this channel should contain
        values = [chan_cfg.value]
        for other_chan, other_cfg in config.channels.items():
            if other_chan != chan and chan in other_cfg.contained_by:
                values.append(other_cfg.value)

        # Isolate the channel volume
        chan_vol = np.isin(seg_volume, values)
        print(f"Channel volume has {np.sum(chan_vol)} voxels")
        io.imsave(f"channel_{chan}.tif", chan_vol.astype(np.uint8) * 255)

        # convert radius to spherical volume
        chan_min_vol = (4/3) * np.pi * (chan_cfg.min_radius ** 3)

        print(f"Initial blur")
        # Apply blurring and thresholding
        if chan_cfg.blur_sigma > 0:
            chan_vol = ndi.gaussian_filter(chan_vol.astype(float), sigma=chan_cfg.blur_sigma)
            chan_vol = chan_vol > 0.5

        print(f"Filling holes and labeling")
        # Fill holes if specified
        if chan_cfg.fill_holes:
            chan_vol = ndi.binary_fill_holes(chan_vol)

        # Apply connectivity-based instance segmentation
        labels, num_labels = measure.label(chan_vol, connectivity=2, return_num=True)

        print(f"Found {num_labels} initial labels")
        # Apply watershed if specified
        if chan_cfg.apply_watershed:
            for i in range(num_labels):
                # Create a mask for the current label
                label_mask = labels == (i + 1)

                # Check if it is large enough
                if np.sum(label_mask) < chan_min_vol:
                    labels[label_mask] = 0
                    continue

                print(f"Processing label {i+1} with volume {np.sum(label_mask)} voxels")
                io.imsave(f"label_{chan}_{i+1}.tif", label_mask.astype(np.uint8) * 255)

                # Extract subvolume to calculate watershed
                binary_subvolume, coords = get_subvolume(label_mask)

                watershed_labels = watershed(binary_subvolume.astype(np.uint8), min_size=chan_cfg.min_radius)
                num_ws_labels = np.unique(watershed_labels).size - 1
                print(f"Label {i+1}: watershed split into {num_ws_labels} labels")
                if num_ws_labels > 1:
                    for j in range(1, num_ws_labels + 1):
                        # Get coords of the current watershed label
                        new_max = np.max(labels) + 1
                        ws_coords = np.where(watershed_labels == j)
                        min_coords = coords[0]
                        translated_coords = tuple(ws_coord + min_coord for ws_coord, min_coord in zip(ws_coords, min_coords))
                        labels[translated_coords] = new_max

        print(f"Smoothing labels")
        # Smooth instances on GPU
        if chan_cfg.smooth_iterations > 0:
            labels = torch.from_numpy(labels).to('cuda')
            labels = smooth_instances_gpu(labels, sigma=1.0, morph_opening=True, opening_iters=chan_cfg.smooth_iterations)
            labels = labels.cpu().numpy()

        seg_out[labels > 0] = chan_cfg.value

    return seg_out


def main():
    args = parse_args()
    
    config = default_config()
    if args.config:
        config_override = omegaconf.OmegaConf.load(args.config)
        config = omegaconf.OmegaConf.merge(config, config_override)

    os.makedirs(args.output_dir, exist_ok=True)

    seg_volume = io.imread(args.input_file)
    seg_out = process_volume(seg_volume, config)

    io.imsave(f"{args.output_dir}/segmented.tif", seg_out)

if __name__ == "__main__":
    import sys
    sys.argv = [
        "post_process.py",
        "-i", "output_image.tif",
        "-o", "output/",
    ]
    main()