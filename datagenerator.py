# This file implements a PyTorch dataset for the data generator.
# We generate yeast phantoms using generate_yeast_phantom and then generate sinograms using reconstruct_gpu.
# We maintain a buffer of items, since phantoms and sinograms are generated independently.

from skimage import io
import random
from typing import Iterable, Iterator, Optional, List, Tuple
import numpy as np
import torch
from glob import glob
from torch.utils.data.dataset import Dataset
from torchvision.transforms.v2.functional import elastic_transform, gaussian_blur
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import IterableDataset
from tifffile import imread, imwrite
import nrrd
from numpy import typing as npt
from tqdm import tqdm
import os
import time

from phantom import (
    create_noisy_volume_torch,
    generate_and_organize,
    get_valid_voxel_mask,
    voxelize_yeast_cells,
    reconstruct_3d
)

# --- Utility Functions ---

def get_valid_voxel_mask_np(volume: npt.NDArray, radius_padding: float = 2.0) -> npt.NDArray:
    """
    Returns a boolean mask indicating which voxels lie within the valid circular (2D) or cylindrical (3D) region.
    The mask is applied across the last two dimensions (H, W), assuming the projection detector defines a circular FOV.

    Args:
        volume (npt.NDArray): A numpy array with (... , H, W) shape, 
        where the last two dimensions represent the height and width of the projection.
        radius_padding (float): Padding to shrink the valid radius slightly for safety.

    Returns:
        npt.NDArray: Boolean mask of same shape as volume, True for valid voxels.
    """

    H = volume.shape[-2]
    W = volume.shape[-1]
    cy, cx = H / 2, W / 2
    radius = min(H, W) / 2 - radius_padding

    y = np.arange(H).reshape(H, 1)
    x = np.arange(W).reshape(1, W)

    # Create a circular (H, W) mask
    mask2d = ((y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2  # shape (H, W)

    return np.broadcast_to(mask2d, volume.shape)

def quantile_normalize(img: npt.NDArray) -> npt.NDArray:
    quantiles = np.sort(np.random.normal(0, 1, np.prod(img.shape)))
    rankings = np.argsort(img.flatten())
    order = np.argsort(rankings)
    return np.take_along_axis(quantiles, order, None).reshape(img.shape).astype(float)

normal_dist = torch.distributions.Normal(0, 1)

def quantile_normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    flat_tensor = tensor.flatten()
    rankings = torch.argsort(flat_tensor)
    order = torch.argsort(rankings)
    quantiles_tensor = normal_dist.sample(flat_tensor.shape).to(tensor.device)
    quantiles_tensor = torch.sort(quantiles_tensor)[0]
    normalized_tensor = torch.take_along_dim(quantiles_tensor, order, dim=0)
    return normalized_tensor.reshape(tensor.shape).to(dtype=tensor.dtype, device=tensor.device)

def normalize(img: npt.NDArray) -> npt.NDArray:
    if np.max(img) - np.min(img) == 0:
        raise ValueError("Image is constant")
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def normalize_tensor(img: torch.Tensor) -> torch.Tensor:
    if torch.max(img) - torch.min(img) == 0:
        print("Constant image, returning original")
        return img
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

def generate_elastic_displacement(alpha: float, sigma: float, size: List[int], device: torch.device | str ='cuda'):
    alphas = [float(alpha), float(alpha)]
    sigmas = [float(sigma), float(sigma)]
    dx = torch.rand([1, 1] + size, device=device) * 2 - 1
    if sigmas[0] > 0.0:
        kx = int(8 * sigmas[0] + 1)
        if kx % 2 == 0:
            kx += 1
        dx = gaussian_blur(dx, [kx, kx], [sigmas[0]])
    dx = dx * alphas[0] / size[0]
    dy = torch.rand([1, 1] + size, device=device) * 2 - 1
    if sigmas[1] > 0.0:
        ky = int(8 * sigmas[1] + 1)
        if ky % 2 == 0:
            ky += 1
        dy = gaussian_blur(dy, [ky, ky], [sigmas[1]])
    dy = dy * alphas[1] / size[1]
    return torch.concat([dx, dy], 1).permute([0, 2, 3, 1])  # 1 x H x W x 2

def elastic_deformation_GPU(image: torch.Tensor, displacement):
    tensor = image.unsqueeze(0)
    return elastic_transform(tensor, displacement).squeeze()

def elastic_deformation(image: np.ndarray, alpha: int, sigma: int, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    if len(shape) == 3:
        dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))
    else:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing='ij')
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    distorted_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distorted_image.reshape(image.shape)

        
# --- Fully 3D Generator ---
class SyntheticGenerator(IterableDataset):
    def __init__(
        self,
        thetas,
        buffer_max_capacity=10,
        resolution=512,
        crop_size=256,
        target_size=96,
        num_cycles=1,
        num_steps=1000,
        include_phantom: bool = False,
        num_cells: Optional[int] = None,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buffer_max_capacity = buffer_max_capacity
        self.resolution = resolution
        self.crop_size = crop_size
        self.target_size = target_size
        self.thetas = thetas
        self.num_cycles = num_cycles
        self.cur_cycle = 0
        self.cur_idx = 0
        self.buffer = []
        self.iter_obj = iter(self.generate())
        self.num_steps = num_steps
        self.num_cells = num_cells
        self.include_phantom = include_phantom
        self.empty = torch.zeros((1, target_size, target_size, target_size), dtype=torch.half)

    def generate_sample(self, n_cells: Optional[int] = None, include_phantom: Optional[bool] = None):
        if n_cells is None:
            n_cells = max(1, int(np.random.normal(2, 1)))
        if include_phantom is None:
            include_phantom = self.include_phantom


        all_cells, all_vacuoles, all_lipid_droplets = generate_and_organize(n_cells)

        # No grads from this point and on
        with torch.no_grad():
            mask_voxels, phantom = voxelize_yeast_cells(
                all_cells, all_vacuoles, all_lipid_droplets
            )
            
            displacement = generate_elastic_displacement(500, 40, [self.resolution, self.resolution], device=self.device)

            phantom = phantom.to(self.device)
            mask_voxels = mask_voxels.to(self.device)
            for i in range(len(phantom)):
                phantom[i] = elastic_deformation_GPU(phantom[i], displacement)
                for j in range(3):
                    mask_voxels[j, i] = elastic_deformation_GPU(mask_voxels[j, i], displacement)

            noisy_images = create_noisy_volume_torch(phantom)

            phantom = normalize_tensor(phantom).cpu()
            mask_voxels = mask_voxels.cpu()


            theta = np.random.choice(self.thetas)
            # noisy_images is XYZ
            recon = reconstruct_3d(noisy_images, theta=theta)
            valid_circle = get_valid_voxel_mask(recon, radius_padding=5.0)
            valid_values = recon[valid_circle]
            normed_values = quantile_normalize_tensor(valid_values)
            recon[valid_circle] = normed_values
            recon[~valid_circle] = 0.0
            recon = recon.cpu()

            # Center crop all 3 volumes
            start = (self.resolution - self.crop_size) // 2
            end = start + self.crop_size
            if include_phantom:
                phantom = phantom[:, start:end, start:end, start:end]

            recon = recon[start:end, start:end, start:end]
            mask_voxels = mask_voxels[:, start:end, start:end, start:end]

            # Rescale to target size
            if self.target_size != self.crop_size:
                if include_phantom:
                    phantom = torch.nn.functional.interpolate(
                        phantom.unsqueeze(0).unsqueeze(0),
                        size=(self.target_size, self.target_size, self.target_size),
                        mode='area'
                    ).squeeze(0).squeeze(0)
                recon = torch.nn.functional.interpolate(
                    recon.unsqueeze(0).unsqueeze(0),
                    size=(self.target_size, self.target_size, self.target_size),
                    mode='area'
                ).squeeze(0).squeeze(0)
                mask_voxels = torch.nn.functional.interpolate(
                    mask_voxels.unsqueeze(0).float(),
                    size=(self.target_size, self.target_size, self.target_size),
                    mode='nearest'
                ).squeeze(0).long()

            mask = torch.zeros((self.target_size, self.target_size, self.target_size), dtype=torch.long)
            mask[mask_voxels[0] > 0] = 1
            mask[mask_voxels[1] > 0] = 2
            mask[mask_voxels[2] > 0] = 3
            if include_phantom:
                self.buffer.append((phantom.half().unsqueeze(0),
                                    recon.half().unsqueeze(0),
                                    mask.unsqueeze(0)))
            else:
                self.buffer.append((recon.half().unsqueeze(0), mask.unsqueeze(0)))


    def populate(self, n_cells: Optional[int] = None, include_phantom: Optional[bool] = None):
        self.generate_sample(n_cells=n_cells, include_phantom=include_phantom)
        self.buffer = self.buffer[-self.buffer_max_capacity:]

    def generate(self):
        while True:
            if len(self.buffer) == 0:
                self.populate()

            # Sample from buffer
            if self.include_phantom:
                phantom, noisy, mask = self.buffer[self.cur_idx % len(self.buffer)]
                yield (phantom, noisy, mask)
            else:
                noisy, mask = self.buffer[self.cur_idx % len(self.buffer)]
                yield (self.empty, noisy, mask)

            # Check if we need to generate new data
            self.cur_idx += 1
            if self.cur_idx >= len(self.buffer):
                self.cur_idx = 0
                self.cur_cycle += 1
                if self.cur_cycle >= self.num_cycles:
                    self.cur_cycle = 0
                    self.populate()

    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return next(self.iter_obj)
    
    def __len__(self):
        return self.num_steps


class TomoSlice:
    def __init__(self, image: torch.Tensor, idx: int, unique_id: str):
        self.data = image
        self.idx = idx
        self.unique_id = unique_id

class AnnotationSlice:
    def __init__(self, annotation: torch.Tensor, slices: List[str]):
        self.data = annotation
        self.slices = slices  # List of unique IDs corresponding to the slices

class AnnotationBuffer:
    def __init__(self):
        self.tomo_slices = {}
        self.annotation_buffer = []

    def add_tomo_slice(self, tomo_slice: TomoSlice):
        self.tomo_slices[tomo_slice.unique_id] = tomo_slice

    def add_annotation_slice(self, annotation_slice: AnnotationSlice):
        self.annotation_buffer.append(annotation_slice)

    def shuffle(self):
        random.shuffle(self.annotation_buffer)

    def drop_slices(self, num: int):
        if num <= 0:
            return

        self.annotation_buffer = self.annotation_buffer[num:]
        self.cleanup()

    def cleanup(self):
        # Remove tomo slices with no references
        referenced_ids = set()
        for ann_slice in self.annotation_buffer:
            referenced_ids.update(ann_slice.slices)
        self.tomo_slices = {uid: ts for uid, ts in self.tomo_slices.items() if uid in referenced_ids}

    def __len__(self):
        return len(self.annotation_buffer)
    
    def get_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns stacked input tomo and corresponding annotation
        ann_slice = self.annotation_buffer[idx]
        slices = [self.tomo_slices[uid].data for uid in ann_slice.slices]
        input_tomo = torch.stack(slices, dim=0)  # Stack along
        return input_tomo, ann_slice.data


class Semi3DDataGenerator(torch.utils.data.IterableDataset):
    def __init__(
        self,
        thetas: List[int],
        buffer_max_capacity: int = 4096,
        size: int = 512,
        padding: int = 3, 
        num_cycles: int = 4,
        num_steps: int = 100,
        include_phantom: bool = False,
        background_filter_rate: float = 0.0,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.buffer_max_capacity = buffer_max_capacity
        self.size = size
        self.include_phantom = include_phantom
        self.padding = padding
        self.thetas = thetas
        self.num_cycles = num_cycles
        self.cur_cycle = 0
        self.cur_idx = 0
        self.num_steps = num_steps
        self.background_filter_rate = background_filter_rate

        self.rand_phantom_idx = []

        # Buffers (CPU)
        self.annotation_buffer: AnnotationBuffer = AnnotationBuffer()
        self.iter_obj = iter(self.generate())


    def generate_sample(self, n_cells: Optional[int] = None, include_phantom: Optional[bool] = None):
        if n_cells is None:
            n_cells = max(1, int(np.random.normal(3, 2)))
        if include_phantom is None:
            include_phantom = self.include_phantom

        all_cells, all_vacuoles, all_lipid_droplets = generate_and_organize(n_cells)

        with torch.no_grad():
            mask_voxels, phantom = voxelize_yeast_cells(
                all_cells, all_vacuoles, all_lipid_droplets
            )

            displacement = generate_elastic_displacement(
                500, 40, [self.size, self.size], device=self.device
            )

            phantom = phantom.to(self.device)
            mask_voxels = mask_voxels.to(self.device)

            for i in range(len(phantom)):
                phantom[i] = elastic_deformation_GPU(phantom[i], displacement)
                for j in range(3):
                    mask_voxels[j, i] = elastic_deformation_GPU(
                        mask_voxels[j, i], displacement
                    )

            noisy_images = create_noisy_volume_torch(phantom)
            phantom = normalize_tensor(phantom).cpu()
            mask_voxels = mask_voxels.cpu()

            theta = np.random.choice(self.thetas)
            recon = reconstruct_3d(noisy_images, theta=theta)

            valid_circle = get_valid_voxel_mask(recon, radius_padding=5.0)
            valid_values = recon[valid_circle]
            recon[valid_circle] = quantile_normalize_tensor(valid_values)
            recon[~valid_circle] = 0.0
            recon = recon.cpu()

            mask = torch.zeros((self.size, self.size, self.size), dtype=torch.long)
            mask[mask_voxels[0] > 0] = 1
            mask[mask_voxels[1] > 0] = 2
            mask[mask_voxels[2] > 0] = 3

            num_slices = recon.shape[0]
            # Generate unique samplename from timestamp and random number
            unique_sample_name = f"sample_{time.time()}_{random.randint(0, 1000)}"
            for idx in range(num_slices):
                unique_id = f"{unique_sample_name}_slice_{idx}"
                tomo_slice = TomoSlice(recon[idx], idx, unique_id)
                self.annotation_buffer.add_tomo_slice(tomo_slice)

            # Add annotation slices
            for idx in range(self.padding, num_slices - self.padding):
                ann_slice = mask[idx]
                if torch.all(ann_slice == 0):
                    # Skip background-only slices based on filter rate
                    if random.random() < self.background_filter_rate:
                        continue
                slice_ids = [f"{unique_sample_name}_slice_{i + idx}" for i in range(-self.padding, self.padding + 1)]
                annotation_slice = AnnotationSlice(ann_slice, slice_ids)
                self.annotation_buffer.add_annotation_slice(annotation_slice)

            self.annotation_buffer.cleanup()



    def populate(self):
        self.generate_sample()
        self.annotation_buffer.drop_slices(len(self.annotation_buffer) - self.buffer_max_capacity)
        self.annotation_buffer.shuffle()

    def generate(self):
        while True:
            while len(self.annotation_buffer) == 0:
                self.populate()

            noisy, masks = self.annotation_buffer.get_pair(self.cur_idx)

            if self.include_phantom:
                raise NotImplementedError("Phantom inclusion not implemented in Semi3DDataGenerator")
            else:
                yield noisy.unsqueeze(0), masks.unsqueeze(0)

            self.cur_idx += 1

            if self.cur_idx >= len(self.annotation_buffer):
                self.cur_idx = 0
                self.cur_cycle += 1

                if self.cur_cycle >= self.num_cycles:
                    self.cur_cycle = 0
                    self.populate()

    def __iter__(self):
        return self.iter_obj

    def __len__(self):
        return self.num_steps

    def __next__(self):
        return next(self.iter_obj)

class ValidationSemi3DDataGenerator(Semi3DDataGenerator):
    def __init__(
        self,
        thetas: List[int],
        size: int = 512,
        padding: int = 3,
    ):
        super().__init__(
            thetas=thetas,
            buffer_max_capacity=1,  # No buffering for validation
            size=size,
            padding=padding,
            num_cycles=1,
            num_steps=1,
            include_phantom=False,
            background_filter_rate=1.0,
        )
    
    # Override populate to find a random sample that maximizes the number of annotation components
    def populate(self):
        self.generate_sample()
        self.annotation_buffer.cleanup()
        # Iterate through and maintain a list of indices with max torch.unique counts
        max_unique = 0
        indices_with_max = []
        for idx in range(len(self.annotation_buffer)):
            _, ann_slice = self.annotation_buffer.get_pair(idx)
            num_unique = len(torch.unique(ann_slice))
            if num_unique > max_unique:
                max_unique = num_unique
                indices_with_max = [idx]
            elif num_unique == max_unique:
                indices_with_max.append(idx)

        # Randomly select one of the indices with max unique counts
        selected_idx = random.choice(indices_with_max)
        selected_ann_slice = self.annotation_buffer.annotation_buffer[selected_idx]
        # Rebuild annotation buffer with only the selected slice
        self.annotation_buffer.annotation_buffer = [selected_ann_slice]
        self.annotation_buffer.cleanup()


# --- Real Data Generator ---
class RealDataGenerator(IterableDataset):
    def __init__(self, data_path: str, crop_size: int=256, target_size: int=96, num_steps=1000):
        self.data_path = data_path
        self.crop_size = crop_size
        self.target_size = target_size
        self.num_steps = num_steps
        self.tomograms, self.annotations = self.load_data()
        self.cur_idx = 0
        self.iter_obj = iter(self.generate())
        # Create an empty tensor for the clean image
        self.empty = torch.zeros((1, target_size, target_size, target_size), dtype=torch.half)

    def load_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        tomograms = []
        annotations = []

        for file in tqdm(glob(f"{self.data_path}/*.tif*"), desc="Loading annotated data"):
            tomo = imread(file)
            seg_file = file.replace('.tif', '.seg.nrrd')
            seg_file = seg_file.replace('.tiff', '.seg.nrrd')
            # if not exist error
            if not glob(seg_file):
                print(f"Segmentation file {seg_file} not found for {file}, skipping.")
                continue
            seg, _ = nrrd.read(seg_file) # read as whc
            seg = np.swapaxes(seg, 2, 0)  # Convert to chw format
            
            tomo_tensor = torch.tensor(-tomo, dtype=torch.half)

            valid_circle = get_valid_voxel_mask(tomo_tensor, radius_padding=5.0)
            valid_values = tomo_tensor[valid_circle]
            normed_values = quantile_normalize_tensor(valid_values)
            tomo_tensor[valid_circle] = normed_values
            tomo_tensor[~valid_circle] = 0.0

            tomo_tensor = tomo_tensor.cpu()

            seg_tensor = torch.from_numpy(seg).long() - 1 # make unlabeled  -1
            
            tomograms.append(tomo_tensor)
            annotations.append(seg_tensor)

        return tomograms, annotations

    def generate(self):
        while True:
            if self.cur_idx >= len(self.tomograms):
                self.cur_idx = 0

            noisy = self.tomograms[self.cur_idx]
            mask = self.annotations[self.cur_idx]

            # Perform random 3D crop of size self.crop_size
            start_d = random.randint(0, noisy.shape[0] - self.crop_size)
            start_h = random.randint(0, noisy.shape[1] - self.crop_size)
            start_w = random.randint(0, noisy.shape[2] - self.crop_size)

            noisy = noisy[start_d:start_d+self.crop_size,
                          start_h:start_h+self.crop_size,
                          start_w:start_w+self.crop_size].unsqueeze(0)
            
            mask = mask[start_d:start_d+self.crop_size,
                        start_h:start_h+self.crop_size,
                        start_w:start_w+self.crop_size].unsqueeze(0)
            
            # Rescale to target size
            if self.target_size != self.crop_size:
                noisy = torch.nn.functional.interpolate(
                    noisy.unsqueeze(0),
                    size=(self.target_size, self.target_size, self.target_size),
                    mode='area'
                ).squeeze(0)
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).float(),
                    size=(self.target_size, self.target_size, self.target_size),
                    mode='nearest'
                ).squeeze(0).long()

            yield (self.empty, noisy, mask)

            self.cur_idx += 1
            

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return next(self.iter_obj)
    
    def __len__(self):
        return self.num_steps


class RealSemi3DDataGenerator(IterableDataset):
    def __init__(self, data_path: str, resolution: int = 512, padding: int = 3):
        self.data_path = data_path
        self.resolution = resolution
        self.padding = padding

        self.annotation_buffer = AnnotationBuffer() 
        self.load_data()

        self.cur_idx = 0
        self.iter_obj = iter(self.generate())

    def load_data(self):

        files = os.listdir(self.data_path)
        raw_files = sorted(f for f in files if f.endswith(".tif") or f.endswith(".tiff"))

        for raw_name in tqdm(raw_files):
            seg_name = raw_name.replace('.tif', '.seg.nrrd').replace('.tiff', '.seg.nrrd')

            raw_path = os.path.join(self.data_path, raw_name)
            seg_path = os.path.join(self.data_path, seg_name)

            if not os.path.exists(seg_path):
                continue

            # Load volumes
            raw = imread(raw_path)
            seg, _ = nrrd.read(seg_path)
            seg = np.swapaxes(seg, 2, 0)  # Convert to chw format

            # Reduce to inner circular region and quantile normalize
            valid_mask = get_valid_voxel_mask_np(raw)
            valid_values = raw[valid_mask]
            normed_values = quantile_normalize(valid_values)
            raw[valid_mask] = normed_values
            raw[~valid_mask] = 0.0
            # Remove annotations outside valid region
            seg[~valid_mask] = 0 

            if raw.ndim != 3:
                raise ValueError(f"{raw_name} is not 3D")

            num_slices = raw.shape[0]
            for idx in range(num_slices):
                unique_id = f"{raw_name}_slice_{idx}"
                tomo_slice = TomoSlice(
                    torch.tensor(raw[idx], dtype=torch.half), 
                    idx, 
                    unique_id
                )
                self.annotation_buffer.add_tomo_slice(tomo_slice)

            for idx in range(self.padding, num_slices - self.padding):
                ann_slice = torch.from_numpy(seg[idx]).long() - 1  # make unlabeled -1
                if torch.all(ann_slice == -1):
                    continue # Unlabeled slice
                slice_ids = [f"{raw_name}_slice_{i + idx}" for i in range(-self.padding, self.padding + 1)]
                annotation_slice = AnnotationSlice(ann_slice, slice_ids)
                self.annotation_buffer.add_annotation_slice(annotation_slice)

            #break # Uncomment to only load one sample for testing
        self.annotation_buffer.cleanup()

            

    def generate(self):
        while True:
            if self.cur_idx >= len(self.annotation_buffer):
                self.cur_idx = 0

            noisy, mask = self.annotation_buffer.get_pair(self.cur_idx)
            self.cur_idx += 1
            if self.cur_idx >= len(self.annotation_buffer):
                self.cur_idx = 0

            yield (noisy.unsqueeze(0), mask.unsqueeze(0))


    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return next(self.iter_obj)
    
    def __len__(self):
        return len(self.annotation_buffer)
                 
    

class MixedDataGenerator(IterableDataset):
    def __init__(self, synthetic_generator, real_data_generator, real_ratio: float=0.5, num_steps=1000):
        self.synthetic_generator = synthetic_generator
        self.real_data_generator = real_data_generator
        self.real_ratio = real_ratio
        self.num_steps = num_steps
        self.iter_obj = iter(self.generate())

    def generate(self):
        while True:
            if random.random() > self.real_ratio:
                # Get from synthetic generator
                noisy, mask = next(self.synthetic_generator)
                yield (noisy, mask)
            else:
                # Get from real data generator
                noisy, mask = next(self.real_data_generator)
                yield (noisy, mask)

    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return next(self.iter_obj)
    
    def __len__(self):
        return self.num_steps