import glob
import time

import numpy as np
import pandas as pd
import omegaconf
import pyvista as pv
import torch
import torch.nn.functional as F
import torch_radon
from random import randint, random, choice, gauss, sample
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar, OptimizeResult
from skimage import io
from skimage.transform import resize

from util import chance, clip
from apollonian_packing import generate_apollonian
from typing import Literal, Optional, Tuple, List, Union, overload

normal_dist = torch.distributions.Normal(0, 1)
def quantile_normalize_tensor(tensor):
    flat_tensor = tensor.flatten()
    rankings = torch.argsort(flat_tensor)
    order = torch.argsort(rankings)
    quantiles_tensor = normal_dist.sample(flat_tensor.shape).to(tensor.device)
    quantiles_tensor = torch.sort(quantiles_tensor)[0]
    normalized_tensor = torch.take_along_dim(quantiles_tensor, order, dim=0)
    return normalized_tensor.reshape(tensor.shape).to(dtype=tensor.dtype, device=tensor.device)

def get_valid_voxel_mask(volume: torch.Tensor, radius_padding: float = 2.0) -> torch.Tensor:
    """
    Returns a boolean mask indicating which voxels lie within the valid circular (2D) or cylindrical (3D) region.
    The mask is applied across the last two dimensions (H, W), assuming the projection detector defines a circular FOV.

    Args:
        volume (torch.Tensor): A tensor with (... , H, W) shape, 
        where the last two dimensions represent the height and width of the projection.
        radius_padding (float): Padding to shrink the valid radius slightly for safety.

    Returns:
        torch.Tensor: Boolean mask of shape (D, 4, H, W), True for valid voxels.
    """

    H = volume.shape[-2]
    W = volume.shape[-1]
    cy, cx = H / 2, W / 2
    radius = min(H, W) / 2 - radius_padding

    y = torch.arange(H, device=volume.device).view(H, 1)
    x = torch.arange(W, device=volume.device).view(1, W)

    # Create a circular (H, W) mask
    mask2d = ((y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2  # shape (H, W)

    return mask2d.expand(volume.shape)


def create_ellipsoid(center: np.ndarray, radii: np.ndarray, resolution: int = 50) -> pv.StructuredGrid:
    """
    Create a 3D ellipsoid mesh using pyvista.

    Args:
        center: Center of the ellipsoid (x, y, z).
        radii: Radii along x, y, z axes.
        resolution: Number of points for mesh generation.

    Returns:
        pv.StructuredGrid: The ellipsoid mesh.
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radii[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radii[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    return pv.StructuredGrid(x, y, z)


def is_point_inside_ellipsoid(point: torch.Tensor, center: torch.Tensor, radii: torch.Tensor) -> bool:
    """
    Check if a point is inside an ellipsoid.

    Args:
        point: The point to check.
        center: Center of the ellipsoid.
        radii: Radii of the ellipsoid.

    Returns:
        bool: True if inside, False otherwise.
    """
    transformed_point = (point - center) / radii
    return torch.sum(transformed_point ** 2).item() <= 1.0


def ellipsoid_grid_points(center: torch.Tensor, radii: torch.Tensor, grid_size: int = 64, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate a 3D boolean grid indicating points inside an ellipsoid.

    Args:
        center: Center of the ellipsoid.
        radii: Radii along x, y, z axes.
        grid_size: Size of the grid along each axis.
        device: Device to create tensor on.

    Returns:
        torch.Tensor: Boolean 3D tensor, True for points inside the ellipsoid.
    """
    if device is None:
        device = center.device
    
    x = torch.linspace(-1, 1, grid_size, device=device) * radii[0] + center[0]
    y = torch.linspace(-1, 1, grid_size, device=device) * radii[1] + center[1]
    z = torch.linspace(-1, 1, grid_size, device=device) * radii[2] + center[2]
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    inside = (((X - center[0]) / radii[0]) ** 2 +
              ((Y - center[1]) / radii[1]) ** 2 +
              ((Z - center[2]) / radii[2]) ** 2) <= 1.0
    return inside


def is_ellipsoid_inside_ellipsoid(
    center_A: torch.Tensor, radii_A: torch.Tensor,
    center_B: torch.Tensor, radii_B: torch.Tensor,
    grid_size: int = 64,
    device: Optional[torch.device] = None
) -> bool:
    """
    Check if ellipsoid A is fully inside ellipsoid B using a 3D grid.

    Args:
        center_A: Center of ellipsoid A.
        radii_A: Radii of ellipsoid A.
        center_B: Center of ellipsoid B.
        radii_B: Radii of ellipsoid B.
        grid_size: Grid resolution.
        device: Device to create tensor on.

    Returns:
        bool: True if A is fully inside B.
    """
    if device is None:
        device = center_A.device
    
    grid_A = ellipsoid_grid_points(center_A, radii_A, grid_size, device)
    coords = torch.where(grid_A)
    x_min, y_min, z_min = torch.min(torch.stack(coords), dim=1)[0] / (grid_size - 1) * 2 - 1
    x_max, y_max, z_max = torch.max(torch.stack(coords), dim=1)[0] / (grid_size - 1) * 2 - 1
    
    x = torch.linspace(x_min, x_max, grid_size, device=device) * radii_A[0] + center_A[0]
    y = torch.linspace(y_min, y_max, grid_size, device=device) * radii_A[1] + center_A[1]
    z = torch.linspace(z_min, z_max, grid_size, device=device) * radii_A[2] + center_A[2]
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    inside_B = (((X - center_B[0]) / radii_B[0]) ** 2 +
                ((Y - center_B[1]) / radii_B[1]) ** 2 +
                ((Z - center_B[2]) / radii_B[2]) ** 2) <= 1.0
    
    return torch.all(inside_B[grid_A]).item() # type: ignore


def get_ellipsoid_matrix(radii: torch.Tensor) -> torch.Tensor:
    """
    Get the ellipsoid matrix from radii.

    Args:
        radii: Radii along x, y, z axes.

    Returns:
        torch.Tensor: Diagonal matrix of squared radii.
    """
    return torch.diag(torch.square(radii))


def ellipsoid_intersection_test(
    Sigma_A: Union[np.ndarray, torch.Tensor],
    Sigma_B: Union[np.ndarray, torch.Tensor],
    mu_A: Union[np.ndarray, torch.Tensor], 
    mu_B: Union[np.ndarray, torch.Tensor], 
    tau: float = 1.0
) -> bool:
    """
    Test intersection between two ellipsoids.

    Args:
        Sigma_A: Covariance matrix of ellipsoid A.
        Sigma_B: Covariance matrix of ellipsoid B.
        mu_A: Center of ellipsoid A.
        mu_B: Center of ellipsoid B.
        tau: Scaling parameter.

    Returns:
        bool: True if intersection exists.
    """
    if isinstance(Sigma_A, torch.Tensor):
        Sigma_A = Sigma_A.cpu().numpy()
    if isinstance(Sigma_B, torch.Tensor):
        Sigma_B = Sigma_B.cpu().numpy()
    if isinstance(mu_A, torch.Tensor):
        mu_A = mu_A.cpu().numpy()
    if isinstance(mu_B, torch.Tensor):
        mu_B = mu_B.cpu().numpy()

    lambdas, Phi = eigh(Sigma_A, b=Sigma_B)
    v_squared = np.dot(Phi.T, mu_A - mu_B) ** 2
    res = minimize_scalar(K_function, bracket=[0.0, 0.5, 1.0], args=(lambdas, v_squared, tau))
    assert isinstance(res, OptimizeResult)
    return (res.fun >= 0)

def K_function(s: float, lambdas: np.ndarray, v_squared: np.ndarray, tau: float) -> float:
    """
    Helper function for ellipsoid intersection.

    Args:
        s: Scalar parameter.
        lambdas: Eigenvalues.
        v_squared: Squared projections.
        tau: Scaling parameter.

    Returns:
        float: Value of the function.
    """
    return 1. - (1. / tau ** 2) * np.sum(v_squared * ((s * (1. - s)) / (1. + s * (lambdas - 1.))))

def ellipsoid_intersection(
    center1: torch.Tensor, radii1: torch.Tensor,
    center2: torch.Tensor, radii2: torch.Tensor, tau: float = 1.0
) -> bool:
    """
    Check if two ellipsoids intersect.

    Args:
        center1: Center of first ellipsoid.
        radii1: Radii of first ellipsoid.
        center2: Center of second ellipsoid.
        radii2: Radii of second ellipsoid.
        tau: Scaling parameter.

    Returns:
        bool: True if intersection exists.
    """
    Sigma_A = get_ellipsoid_matrix(radii1)
    Sigma_B = get_ellipsoid_matrix(radii2)
    return ellipsoid_intersection_test(Sigma_A, Sigma_B, center1, center2, tau)


def vac_radii_from_cell(volume: float) -> torch.Tensor:
    """
    Calculate vacuole radii given cell volume.

    Args:
        volume: Volume of the cell in voxels.

    Returns:
        np.ndarray: Radii of the vacuole.
    """
    vac_volume = volume / 3.5
    vac_radius = (3 * vac_volume / (4 * np.pi)) ** (1 / 3)
    return torch.normal(vac_radius, vac_radius * 0.1, size=(3,))


def has_lipid_collisions(
    droplet_center: torch.Tensor, droplet_radius: float,
    lipid_droplets: List[Tuple[torch.Tensor, float]], epsilon: float = 1e-2
) -> bool:
    """
    Check if a lipid droplet collides with others.

    Args:
        droplet_center: Center of the droplet.
        droplet_radius: Radius of the droplet.
        lipid_droplets: List of existing droplets (center, radius).
        epsilon: Small buffer.

    Returns:
        bool: True if collision detected.
    """
    for other_center, other_radius in lipid_droplets:
        distance = torch.norm(droplet_center - other_center)
        min_distance = droplet_radius + other_radius + epsilon
        if distance < min_distance:
            return True
    return False


def has_vac_collisions(
    droplet_center: torch.Tensor, droplet_radius: float,
    vacuoles: List[Tuple[torch.Tensor, torch.Tensor]], fast_check: bool = True
) -> bool:
    """
    Check if a droplet collides with any vacuole.

    Args:
        droplet_center: Center of the droplet.
        droplet_radius: Radius of the droplet.
        vacuoles: List of vacuoles (center, radii).
        fast_check: If True, only check closest vacuole.

    Returns:
        bool: True if collision detected.
    """
    radii = torch.tensor([droplet_radius] * 3)
    if fast_check:
        closest_vac = None
        closest_distance = torch.tensor(float('inf'))
        for other_center, other_radii in vacuoles:
            distance = torch.norm(droplet_center - other_center)
            if distance < closest_distance:
                closest_distance = distance
                closest_vac = (other_center, other_radii)
        if closest_vac is not None:
            return (ellipsoid_intersection(droplet_center, radii, closest_vac[0], closest_vac[1]) and
                    not is_ellipsoid_inside_ellipsoid(droplet_center, radii, closest_vac[0], closest_vac[1]))
    for other_center, other_radii in vacuoles:
        if (ellipsoid_intersection(droplet_center, radii, other_center, other_radii) and
                not is_ellipsoid_inside_ellipsoid(droplet_center, radii, other_center, other_radii)):
            return True
    return False


def draw_ellipsoid_in_array(
    shape: Tuple[int, int, int], center: torch.Tensor, radii: torch.Tensor,
    origin: Union[Tuple[int, int, int], torch.Tensor], scale: float = 1
) -> torch.Tensor:
    """
    Draw an ellipsoid in a 3D array, optimized to only compute within the bounding box.

    Args:
        shape: Shape of the array.
        center: Center of the ellipsoid.
        radii: Radii of the ellipsoid.
        origin: Origin of the array.
        scale: Scaling factor.

    Returns:
        torch.Tensor: Boolean mask of the ellipsoid.
    """
    if isinstance(origin, tuple):
        origin = torch.tensor(origin, dtype=torch.float32, device=center.device)
    scaled_center = center * scale + origin
    scaled_radii = radii * scale
    x_c, y_c, z_c = scaled_center
    a, b, c = scaled_radii
    nx, ny, nz = shape

    # Compute bounding box, clipped to array bounds
    x_min = max(int(torch.floor(x_c - a).item()), 0)
    x_max = min(int(torch.ceil(x_c + a).item()) + 1, nx)
    y_min = max(int(torch.floor(y_c - b).item()), 0)
    y_max = min(int(torch.ceil(y_c + b).item()) + 1, ny)
    z_min = max(int(torch.floor(z_c - c).item()), 0)
    z_max = min(int(torch.ceil(z_c + c).item()) + 1, nz)

    # Create local grid
    x, y, z = torch.meshgrid(
        torch.arange(x_min, x_max, device=center.device),
        torch.arange(y_min, y_max, device=center.device),
        torch.arange(z_min, z_max, device=center.device),
        indexing='ij'
    )
    mask_local = ((x - x_c) ** 2 / a ** 2 +
                  (y - y_c) ** 2 / b ** 2 +
                  (z - z_c) ** 2 / c ** 2) <= 1

    # Place local mask into full array
    mask = torch.zeros(shape, dtype=torch.bool, device=center.device)
    mask[x_min:x_max, y_min:y_max, z_min:z_max] = mask_local
    return mask

def get_default_config() -> omegaconf.DictConfig:
    """
    Get the default configuration for the phantom generation.

    Returns:
        omegaconf.DictConfig: Default configuration.
    """
    return omegaconf.OmegaConf.create({
        'cell_membrane': {
            'lac_mean': 0.346,
            'lac_std': 0.027,
            'radius_mean': 50.0,
            'std_radius': 2.5,
            'radius_min': 40.0,
            'radius_max': 60.0,
        },
        'vacuole': {
            'lac_mean': 0.182,
            'lac_std': 0.039,
            'radius_mean':38.0,
            'std_radius': 8.0,
            'radius_min': 25.0,
            'radius_max': 45.0,
            'fracture_rate': 0.3,
        },
        'lipid_droplet': {
            'lac_mean': 0.570,
            'lac_std': 0.111,
            'radius_mean': 6.0,
            'std_radius': 1,
            'radius_min': 5.0,
            'radius_max': 8.0,
            'n_mean': 15,
            'n_std': 5,
            'n_min': 3,
            'n_max': 30,
        },
        'fiducial_marker': {
            'radius': 4.0,
            'lac': 2.0,
            'n_mean': 16,
            'n_std': 4,
            'n_min': 3,
            'n_max': 25,
        },
        'noise_parameters': {
            'ice_cracks': {
                'lac_mean': 0.2,
                'lac_std': 0.02,
                'lac_min': 0.05,
                'lac_max': 0.5,
                'probability': 1.0,
            },
            'perlin_noise': {
                'mean': 0.1,
                'std': 0.02,
                'min': 0.0,
                'max': 0.1,
            },
            'flicker_effect': {
                'mean': 0.05,
                'std': 0.03,
                'min': 0.0,
                'max': 1.0,
                'probability': 1.0,
            },    
            'gaussian_blur': {
                'sigma_mean': 3.0,
                'sigma_std': 2.0,
                'sigma_min': 1.0,
                'sigma_max': 5.0,
                'probability': 1.0,
            },
            'gaussian_noise': {
                'mu_mean': 0.0001,
                'mu_std': 0.0,
                'sigma_mean': 0.001,
                'sigma_std': 0.001,
                'sigma_min': 0.001,
                'probability': 1.0,
            },
            'missing_projection': {
                'probability': 0.20,
            },
            'ice_thickness': {
                'mean': 10000.0,
                'std': 2000.0,
                'min': 5000.0,
                'max': 20000.0,
                'lac_mean': 0.01,
                'lac_std':  0.00001,
            }
        },
    })


def generate_yeast_cell(
    scale: float = 1.0,
    origin: Tuple[int, int, int] | torch.Tensor = (0, 0, 0),
    config: Optional[omegaconf.DictConfig] = None,
    device: Optional[torch.device] = None):
    """
    Generate a synthetic yeast cell with vacuoles and lipid droplets.

    Args:
        scale: Scaling factor.
        origin: Origin of the cell.
        config: Configuration dictionary.
        device: Device to use for tensors.

    Returns:
        Tuple of cell membranes, vacuoles, and lipid droplets.
    """
    if config is None:
        config = get_default_config()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(origin, tuple):
        origin = torch.tensor(origin, dtype=torch.float32, device=device)

    # Cell membrane
    cell_mean_radius = config.cell_membrane.radius_mean * scale
    cell_std_radius = config.cell_membrane.std_radius * scale
    cell_radii = torch.clamp(
        torch.normal(cell_mean_radius, cell_std_radius, size=(3,), device=device),
        min=config.cell_membrane.radius_min * scale,
        max=config.cell_membrane.radius_max * scale
    )
    vacuoles: List[Tuple[torch.Tensor, torch.Tensor]] = []
    num_vacuoles = int(torch.clamp(
        torch.normal(2.0, 2.0, size=(1,), device=device),
        min=torch.tensor(1.0, device=device),
        max=torch.tensor(4.0, device=device)
    ).item())
    
    for _ in range(num_vacuoles):
        # Sample vac radius, then we allow some variance across the three axes
        vac_rad = torch.normal(config.vacuole.radius_mean * scale, config.vacuole.std_radius * scale, size=(1,), device=device)
        vac_rad /= num_vacuoles  # Scale down by number of vacuoles
        vac_radii = torch.clamp(
            torch.normal(vac_rad.item(), vac_rad.item() * 0.05, size=(3,), device=device),
            min=config.vacuole.radius_min * scale,
            max=config.vacuole.radius_max * scale
        )
        max_vacuole_displacement = cell_radii - vac_radii
        for _ in range(25):
            vac_center = origin + (torch.rand(3, device=device) * 2 - 1) * max_vacuole_displacement
            if is_ellipsoid_inside_ellipsoid(vac_center, vac_radii, origin, cell_radii * 0.95) and \
                    not any(ellipsoid_intersection(vac_center, vac_radii, other_center, other_radii)
                            for other_center, other_radii in vacuoles):
                vacuoles.append((vac_center, vac_radii))
                break
    
    fractured_vacuoles: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for vac_center, vac_radii in vacuoles:
        if chance(config.vacuole.fracture_rate):
            centers, radii = generate_apollonian(depth=1)
            centers = torch.tensor(centers, dtype=torch.float32, device=device)
            radii = torch.tensor(radii, dtype=torch.float32, device=device)
            volume = (4 / 3) * np.pi * torch.prod(vac_radii).item()
            volumes = [4 / 3 * np.pi * r.item() ** 3 for r in radii]
            scaler = (volume / sum(volumes)) ** (1/3) * 0.9
            radii *= scaler
            centers *= scaler
            fractured_vacuoles.extend(
                [(vac_center + p, torch.full((3,), r.item(), device=device)) 
                 for p, r in zip(centers, radii) if chance(0.7)])
        else:
            fractured_vacuoles.append((vac_center, vac_radii))
    
    lipid_droplets: List[Tuple[torch.Tensor, float]] = []
    for i in reversed(range(len(fractured_vacuoles))):
        vac_center, vac_radii = fractured_vacuoles[i]
        if torch.mean(vac_radii).item() < config.lipid_droplet.radius_max * scale and chance(0.3):
            lipid_droplets.append((vac_center, torch.mean(vac_radii).item()))
            del fractured_vacuoles[i]
    
    n_droplets = int(torch.clamp(
        torch.normal(config.lipid_droplet.n_mean,
                    config.lipid_droplet.n_std, size=(1,), device=device),
        min=torch.tensor(config.lipid_droplet.n_min, dtype=torch.float32, device=device),
        max=torch.tensor(config.lipid_droplet.n_max, dtype=torch.float32, device=device)
    ).item())
    
    for _ in range(n_droplets):
        droplet_radius = torch.clamp(
            torch.normal(config.lipid_droplet.radius_mean, config.lipid_droplet.std_radius, size=(1,), device=device),
            min=config.lipid_droplet.radius_min,
            max=config.lipid_droplet.radius_max
        ).item() * scale
        for _ in range(100):
            ranpos = torch.rand(3, device=device) * 2 - 1
            droplet_center = origin + ranpos * (torch.min(cell_radii).item() - droplet_radius)
            if is_ellipsoid_inside_ellipsoid(droplet_center, torch.full((3,), droplet_radius, device=device), origin, cell_radii) and \
                    not has_lipid_collisions(droplet_center, droplet_radius, lipid_droplets) and \
                    not has_vac_collisions(droplet_center, droplet_radius, fractured_vacuoles):
                lipid_droplets.append((droplet_center, droplet_radius))
                break
    return [(origin, cell_radii)], fractured_vacuoles, lipid_droplets


def visualize_yeast_cells(
    cell_membranes: List[Tuple[np.ndarray, np.ndarray]],
    vacuoles: List[Tuple[np.ndarray, np.ndarray]],
    lipid_droplets: List[Tuple[np.ndarray, float]]
) -> None:
    """
    Visualize yeast cells, vacuoles, and lipid droplets using pyvista.

    Args:
        cell_membranes: List of cell membrane ellipsoids.
        vacuoles: List of vacuole ellipsoids.
        lipid_droplets: List of lipid droplets (center, radius).
    """
    plotter = pv.Plotter()
    for cell_center, cell_radii in cell_membranes:
        cell_mesh = create_ellipsoid(cell_center, cell_radii)
        plotter.add_mesh(cell_mesh, color='white', opacity=0.4)
    for vac_center, vac_radii in vacuoles:
        vacuole_mesh = create_ellipsoid(vac_center, vac_radii)
        plotter.add_mesh(vacuole_mesh, color='red', opacity=0.7)
    for droplet_center, droplet_radius in lipid_droplets:
        droplet_mesh = pv.Sphere(radius=droplet_radius, center=droplet_center)
        plotter.add_mesh(droplet_mesh, color='green', opacity=0.85)
    plotter.show_axes()  # type: ignore
    plotter.show_grid()  # type: ignore
    plotter.show()


def voxelize_yeast_cells(
    cell_membranes: List[Tuple[torch.Tensor, torch.Tensor]],
    vacuoles: List[Tuple[torch.Tensor, torch.Tensor]],
    lipid_droplets: List[Tuple[torch.Tensor, float]],
    resolution: Optional[Tuple[int, int, int]] = None,
    scale: float = 1.2,
    config: Optional[omegaconf.DictConfig] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Voxelize yeast cells, vacuoles, and lipid droplets into a 3D array.

    Args:
        cell_membranes: List of cell membrane ellipsoids.
        vacuoles: List of vacuole ellipsoids.
        lipid_droplets: List of lipid droplets (center, radius).
        array_origin: Origin of the array.
        resolution: Resolution of the array.
        as_tensor: If True, return torch.Tensor, else np.ndarray.
        config: Configuration dictionary.

    Returns:
        Tuple of (mask, phantom).
    """
    if config is None:
        config = get_default_config()
    if resolution is None:
        resolution = (512, 512, 512)

    array_origin = torch.tensor(resolution) // 2
    array_origin = tuple(array_origin.tolist())
    
    voxels = torch.zeros((3, *resolution), dtype=torch.bool)
    phantom = torch.zeros(resolution, dtype=torch.float32)
    
    for cell_center, cell_radii in cell_membranes:
        cell_border_width = 1 + abs(random() * 0.5)
        mask = draw_ellipsoid_in_array(resolution, cell_center, cell_radii + cell_border_width, array_origin, scale)
        voxels[0][mask] = True
        membrane_intensity = torch.normal(config.cell_membrane.lac_mean, torch.tensor(config.cell_membrane.lac_std, device=cell_center.device)).item()
        border_intensity = torch.normal(config.lipid_droplet.lac_mean, torch.tensor(config.lipid_droplet.lac_std, device=cell_center.device)).item()
        phantom[mask] = border_intensity
        mask = draw_ellipsoid_in_array(resolution, cell_center, cell_radii, array_origin, scale)
        voxels[0][mask] = True
        phantom[mask] = membrane_intensity
    
    for vac_center, vac_radii in vacuoles:
        mask = draw_ellipsoid_in_array(resolution, vac_center, vac_radii, array_origin, scale)
        voxels[1][mask] = True
        vac_intensity = torch.normal(config.vacuole.lac_mean, torch.tensor(config.vacuole.lac_std, device=vac_center.device)).item()
        phantom[mask] = vac_intensity
    
    for droplet_center, droplet_radius in lipid_droplets:
        mask = draw_ellipsoid_in_array(resolution, droplet_center, torch.full((3,), droplet_radius, device=droplet_center.device), array_origin, scale)
        voxels[2][mask] = True
        droplet_intensity = torch.abs(torch.normal(config.lipid_droplet.lac_mean, torch.tensor(config.lipid_droplet.lac_std, device=droplet_center.device))).item()
        phantom[mask] = droplet_intensity
    
    return voxels, phantom


def organization_loss(
    pos: torch.Tensor, radii: torch.Tensor, eps: float = 1e-2
) -> Tuple[torch.Tensor, bool]:
    """
    Calculate loss for sphere organization.

    Args:
        pos: Positions of ellipsoids.
        radii: Radii of ellipsoids.
        eps: Small value for overlap.

    Returns:
        Tuple of loss and overlap flag.
    """
    loss = torch.tensor(0.0, device=pos.device)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            distance = torch.linalg.norm(pos[i] - pos[j])
            min_distance = torch.max(radii[i]) + torch.max(radii[j])
            gap = min_distance - distance + eps
            if gap > 1:
                overlap_penalty = gap ** 2
            else:
                overlap_penalty = torch.relu(gap)
            loss += overlap_penalty
    compactness_penalty = torch.sum(torch.norm(pos, dim=1))
    return loss + 0.1 * compactness_penalty, loss.item() > len(pos) * eps


def organize_ellipsoids(
    positions: torch.Tensor, radii: torch.Tensor, max_iter: int = 5000
) -> torch.Tensor:
    """
    Organize ellipsoids to minimize overlap.

    Args:
        positions: Initial positions.
        radii: Radii of ellipsoids.
        max_iter: Maximum iterations.

    Returns:
        torch.Tensor: Optimized positions.
    """
    optimizer = torch.optim.Adam([positions], lr=0.05)
    for _ in range(max_iter):
        optimizer.zero_grad()
        loss, overlap = organization_loss(positions, radii)
        loss.backward()
        optimizer.step()
        if not overlap:
            break
    return positions


def generate_and_organize(
    n_cells: int = 1, z_offset_sigma: float = 0.0
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, float]]]:
    """
    Generate and organize multiple yeast cells.

    Args:
        n_cells: Number of cells.
        z_offset_sigma: Standard deviation for z-offset.

    Returns:
        Tuple of all cells, vacuoles, and lipid droplets.
    """
    device = torch.device('cpu')
    cell_list, vacuoles_list, lipid_droplets_list = [], [], []
    for _ in range(n_cells):
        cell_membranes, vacuoles, lipid_droplets = generate_yeast_cell(device=device)
        cell_list.append(cell_membranes)
        vacuoles_list.append(vacuoles)
        lipid_droplets_list.append(lipid_droplets)
    pos, radii = zip(*[(center, radius) for cell in cell_list for center, radius in cell])
    pos_tensor = torch.stack(pos).to(device).requires_grad_(True)
    radii_tensor = torch.stack(radii).to(device)
    with torch.no_grad():
        pos_tensor[:, :2] += torch.randn_like(pos_tensor[:, :2]) * 10
    organized_pos = organize_ellipsoids(pos_tensor, radii_tensor).detach().cpu()
    all_cells, all_vacuoles, all_lipid_droplets = [], [], []
    for i in range(n_cells):
        offset = organized_pos[i] + torch.tensor([0, 0, random() * z_offset_sigma])
        all_cells.extend([(center + offset, radius) for center, radius in cell_list[i]])
        all_vacuoles.extend([(center + offset, radius) for center, radius in vacuoles_list[i]])
        all_lipid_droplets.extend([(center + offset, radius) for center, radius in lipid_droplets_list[i]])
    return all_cells, all_vacuoles, all_lipid_droplets


def create_noisy_volume_torch(
    vol: torch.Tensor,
    scale: float = 1.2,
    config: Optional[omegaconf.DictConfig] = None,
) -> torch.Tensor:
    """
    Add noise, cracks, and fiducial markers to a 3D volume.

    Args:
        vol: Input volume.
        blur: Gaussian blur sigma.
        crack_prob: Probability of adding cracks.
        crack_scale: Crack intensity scale.
        perlin_noise: Perlin noise intensity.
        fiducial_markers: Number of fiducial markers.
        seed: Random seed.

    Returns:
        torch.Tensor: Noisy volume.
    """

    if config is None:
        config = get_default_config()

    mask = (vol == 0)
    noisy = vol.clone()
    d, h, w = vol.shape

    # Blur the projections a bit to simulate the detector blur
    blur_prob = config.noise_parameters.gaussian_blur.probability
    if blur_prob > 0 and chance(blur_prob):
        sigma = clip(
            gauss(
                config.noise_parameters.gaussian_blur.sigma_mean,
                config.noise_parameters.gaussian_blur.sigma_std
            ),
            config.noise_parameters.gaussian_blur.sigma_min,
            config.noise_parameters.gaussian_blur.sigma_max
        )
        if sigma > 0:
            noisy = apply_gaussian_blur_torch(noisy, sigma)

    valid_coords = torch.nonzero(mask, as_tuple=False).to(vol.device)
    num_valid_coords = valid_coords.shape[0]

    num_fiducial_markers = int(
        clip(
            gauss(config.fiducial_marker.n_mean, config.fiducial_marker.n_std),
            config.fiducial_marker.n_min,
            config.fiducial_marker.n_max
        )
    )

    for _ in range(num_fiducial_markers):
        random_idx = randint(0, num_valid_coords - 1)
        z, y, x = valid_coords[random_idx].squeeze()
        z , y, x = int(z.item()), int(y.item()), int(x.item())
        sphere_rad = config.fiducial_marker.radius * scale
        zz, yy, xx = generate_sphere_torch(sphere_rad, (z,y,x), (d, h, w), device=vol.device)
        zz, yy, xx = zz.to(vol.device), yy.to(vol.device), xx.to(vol.device)
        zz = torch.clamp(zz, 0, d - 1)
        yy = torch.clamp(yy, 0, h - 1)
        xx = torch.clamp(xx, 0, w - 1)
        zz, yy, xx = zz.long(), yy.long(), xx.long()
        noisy[zz, yy, xx] = config.fiducial_marker.lac
    
    crack_prob = config.noise_parameters.ice_cracks.probability
    crack_scale = clip(
        gauss(config.noise_parameters.ice_cracks.lac_mean, config.noise_parameters.ice_cracks.lac_std),
        config.noise_parameters.ice_cracks.lac_min,
        config.noise_parameters.ice_cracks.lac_max
    )
    if crack_prob > 0 and chance(crack_prob):
        perlin_map = io.imread(choice(glob.glob("../LATS/perlin3d_lo/*.tif"))).astype(np.float32)
        if perlin_map.shape != vol.shape:
            perlin_map = resize(perlin_map, vol.shape)
        perlin_map = torch.tensor(perlin_map, dtype=torch.float32).to(vol.device)
        crackmap = io.imread(choice(glob.glob("../LATS/cracks3d/combined_crack_map_*.tif")))
        if crackmap.shape != vol.shape:
            crackmap = resize(crackmap, vol.shape)
        crackmap = torch.tensor(crackmap, dtype=torch.float32).to(vol.device)
        crackmap = (crackmap - crackmap.min()) / (crackmap.max() - crackmap.min())
        perlin_map_normalized = (perlin_map - perlin_map.min()) / (perlin_map.max() - perlin_map.min())
        crackmap *= perlin_map_normalized
        noisy[mask] += crackmap[mask] * crack_scale

    perlin_noise = clip(
        gauss(
            config.noise_parameters.perlin_noise.mean,
            config.noise_parameters.perlin_noise.std
        ),
        config.noise_parameters.perlin_noise.min,
        config.noise_parameters.perlin_noise.max
    )
    if perlin_noise > 0:
        perlin_map = io.imread(choice(glob.glob("../LATS/perlin3d_hi/*.tif"))).astype(np.float32)
        if perlin_map.shape != vol.shape:
            perlin_map = resize(perlin_map, vol.shape)
        perlin_map = torch.tensor(perlin_map, dtype=torch.float32).to(vol.device)
        perlin_map -= perlin_map.min()
        perlin_map /= perlin_map.max()
        # Clip the perlin map to a cylinder to avoid artifacts due to volume being a cube
        circle_mask = get_valid_voxel_mask(vol)
        perlin_map *= circle_mask
        noisy[mask] += perlin_map[mask] * perlin_noise
    
    return noisy


def generate_ellipsoid_torch(
    a: int, b: int, c: int, shape: Tuple[int, int, int]
) -> Tuple[torch.Tensor, ...]:
    """
    Generate a 3D ellipsoid mask.

    Args:
        a: Semi-axis along x.
        b: Semi-axis along y.
        c: Semi-axis along z.
        shape: Output shape.

    Returns:
        Tuple of torch.Tensor: Indices of ellipsoid voxels.
    """
    z, y, x = torch.meshgrid(
        torch.arange(0, shape[0], dtype=torch.float32),
        torch.arange(0, shape[1], dtype=torch.float32),
        torch.arange(0, shape[2], dtype=torch.float32),
    )
    zc, yc, xc = shape[0] // 2, shape[1] // 2, shape[2] // 2
    ellipsoid = (((x - xc) / a) ** 2 + ((y - yc) / b) ** 2 + ((z - zc) / c) ** 2) <= 1
    return torch.nonzero(ellipsoid, as_tuple=True)


def generate_sphere_torch(
    radius: int, center: Tuple[int, int, int], shape: Tuple[int, int, int], device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, ...]:
    """
    Generate a 3D sphere mask.

    Args:
        radius: Sphere radius.
        center: Center of the sphere (z, y, x).
        shape: Output shape.

    Returns:
        Tuple of torch.Tensor: Indices of sphere voxels.
    """
    if device is None:
        device = torch.device('cpu')

    cz, cy, cx = center
    sz, sy, sx = shape

    # Compute bounding box for the sphere, clipped to array bounds
    z_min = max(cz - radius, 0)
    z_max = min(cz + radius + 1, sz)
    y_min = max(cy - radius, 0)
    y_max = min(cy + radius + 1, sy)
    x_min = max(cx - radius, 0)
    x_max = min(cx + radius + 1, sx)

    # Create local grid
    z, y, x = torch.meshgrid(
        torch.arange(z_min, z_max, dtype=torch.float32, device=device),
        torch.arange(y_min, y_max, dtype=torch.float32, device=device),
        torch.arange(x_min, x_max, dtype=torch.float32, device=device),
        indexing="ij"
    )

    # Compute mask for sphere centered at (cz, cy, cx)
    sphere = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) <= radius ** 2
    local_indices = torch.nonzero(sphere, as_tuple=True)
    # Map local indices back to global indices
    global_indices = (
        local_indices[0] + z_min,
        local_indices[1] + y_min,
        local_indices[2] + x_min,
    )
    return global_indices


def create_gaussian_kernel_3d(kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Create a 3D Gaussian kernel.

    Args:
        kernel_size: Size of the kernel.
        sigma: Standard deviation.

    Returns:
        torch.Tensor: 3D Gaussian kernel.
    """
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
    kernel = torch.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def apply_gaussian_blur_torch(
    vol: torch.Tensor, sigma: float, kernel_size: int = 5
) -> torch.Tensor:
    """
    Apply 3D Gaussian blur to a volume.

    Args:
        vol: Input volume.
        sigma: Standard deviation.
        kernel_size: Kernel size.

    Returns:
        torch.Tensor: Blurred volume.
    """
    kernel = create_gaussian_kernel_3d(kernel_size, sigma).unsqueeze(0).unsqueeze(0)
    kernel = kernel.to(vol.device)
    vol = vol.unsqueeze(0).unsqueeze(0)
    # Pad the volume with mean values to avoid edge effects
    padding = kernel_size // 2
    padded_vol = torch.nn.functional.pad(vol, (padding, padding, padding, padding, padding, padding), mode='replicate')
    blurred_vol = torch.nn.functional.conv3d(padded_vol, kernel, padding=0)
    return blurred_vol.squeeze(0).squeeze(0)

@overload
def reconstruct_3d( # type: ignore
    img: torch.Tensor, 
    theta: int = 180, 
    return_sino: Literal[False] = False,
    config: Optional[omegaconf.DictConfig] = None,
    clip_input_to_circle: bool = True
) -> torch.Tensor: ...

@overload
def reconstruct_3d(
    img: torch.Tensor, 
    theta: int = 180, 
    return_sino: Literal[True] = True,
    config: Optional[omegaconf.DictConfig] = None,
    clip_input_to_circle: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

@torch.no_grad()
def reconstruct_3d(
    img: torch.Tensor, 
    theta: int = 180, 
    return_sino: bool = False,
    config: Optional[omegaconf.DictConfig] = None,
    clip_input_to_circle: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Reconstruct a 3D image using Radon transform and simulate various effects.

    Args:
        img: Input image.
        theta: Angle range for projection.
        return_sino: If True, also return sinogram and filtered sinogram.
        noise: Noise level.
        misalignment_sigma: Misalignment noise.
        missing_projection: Fraction of missing projections.
        ice_effect: Ice effect intensity.
        flicker_effect: Flicker effect intensity.
        seed: Random seed.

    Returns:
        torch.Tensor or tuple: Reconstructed image (and optionally sinogram, filtered sinogram).
    """
    if clip_input_to_circle:
        sphere_mask = get_valid_voxel_mask(img)
        img = img * sphere_mask  # Mask to circle

    if config is None:
        config = get_default_config()

    thetas = torch.arange(-theta / 2, theta / 2, 1, dtype=torch.float32)
    # Create offset in angles tomograms (if it is limited angle tomography)
    if theta < 165:
        offset = clip(
            gauss(0.0, 5.0),
            -15, 15
        )
        thetas = (thetas + offset)

    missing_projection = config.noise_parameters.missing_projection.probability
    if missing_projection > 0:
        num_missing = int(len(thetas) * missing_projection)
        missing_indices = sample(range(len(thetas)), num_missing)
        thetas = torch.tensor([t for i, t in enumerate(thetas) if i not in missing_indices], dtype=thetas.dtype)
        thetas_missing = thetas
    else:
        thetas_missing = thetas
    thetas_noise = thetas_missing + 0.1 * torch.normal(torch.zeros_like(thetas_missing), torch.ones_like(thetas_missing)) # Add small noise to angles
    
    radon = torch_radon.Radon(img.shape[0], torch.deg2rad(thetas_noise))
    radon_clean = torch_radon.Radon(img.shape[0], torch.deg2rad(thetas_missing))
    img = img.cuda() # Must be cuda tensor for torch_radon
    sino: torch.Tensor = radon.forward(img) # type: ignore
    projection_shape = sino[:, 0].shape

    # Flat field correction (or the opposite)
    flicker_effect = clip(
        gauss(
            config.noise_parameters.flicker_effect.mean,
            config.noise_parameters.flicker_effect.std
        ),
        config.noise_parameters.flicker_effect.min,
        config.noise_parameters.flicker_effect.max
    )
    if flicker_effect > 0:
        for i in range(sino.shape[1]):
            perlinmap = io.imread(choice(glob.glob("../LATS/perlinmaps_lo/*.tif"))).astype(np.float32)
            perlinmap = (perlinmap - perlinmap.min()) / (perlinmap.max() - perlinmap.min())
            perlinmap = resize(perlinmap, projection_shape)
            perlinmap = torch.tensor(perlinmap, dtype=torch.float32).to(sino.device)
            mean_signal = sino[:, i].mean()
            sino[:, i] += perlinmap * flicker_effect * mean_signal

    # We have to penetrate more ice at when not perpendicular to the ice surface.
    ice_thickness = clip(
        gauss(
            config.noise_parameters.ice_thickness.mean,
            config.noise_parameters.ice_thickness.std
        ),
        config.noise_parameters.ice_thickness.min,
        config.noise_parameters.ice_thickness.max
    )
    if ice_thickness > 0:
        # We have an ice thickness for which more is penetrated at different angles.
        # If we call the thickness h and the angle theta, then the penetrated ice 'p' is given by:
        # h / cos(abs(theta)) = p by using simple trigonometry.
        
        for i, ang in enumerate(thetas_noise):
            penetrated_ice = ice_thickness / torch.cos(torch.deg2rad(torch.abs(ang)))
            absorption = torch.normal(
                mean=config.noise_parameters.ice_thickness.lac_mean,
                std=config.noise_parameters.ice_thickness.lac_std,
                size=sino[:, i].shape,
                device=sino.device
            )
            sino[:, i] += penetrated_ice * absorption

    # sino = sino.max() - sino  # Invert the sinogram

    filtered = filter_sinogram_3d(sino, filter_type='hamming')
    # TODO: clip to circle
    if return_sino:
        return radon_clean.backprojection(filtered), sino, filtered # type: ignore
    else:
        return radon_clean.backprojection(filtered) # type: ignore


def filter_sinogram_3d(
    sinogram: torch.Tensor, filter_type: Literal['ramp', 'hamming'] = 'hamming',
    hamming_start_freq: float = 0.0
) -> torch.Tensor:
    """
    Filter a 3D sinogram.

    Args:
        sinogram: 3D sinogram (num_projections, height, width).
        filter_type: 'ramp' or 'hamming'.
        hamming_start_freq: Start frequency for Hamming filter.

    Returns:
        torch.Tensor: Filtered sinogram.
    """
    num_projections, _, width = sinogram.shape
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * width))))
    pad_width = projection_size_padded - width
    sinogram_padded = F.pad(sinogram, (0, pad_width, 0, 0, 0, 0), mode='constant', value=0)
    size = projection_size_padded
    n = torch.concat((
        torch.arange(1, size // 2 + 1, 2, device=sinogram.device),
        torch.arange(size // 2 - 1, 0, -2, device=sinogram.device)
    ))
    f = torch.zeros(size, device=sinogram.device)
    f[0] = 0.25
    f[1::2] = -1 / (torch.pi * n) ** 2
    ramp_filter = 2 * torch.real(torch.fft.fft(f))
    if filter_type == 'ramp':
        filter = ramp_filter
    elif filter_type == 'hamming':
        hamming_window = torch.hamming_window(size, periodic=True, alpha=0.54, beta=0.46, device=sinogram.device)
        start_index = int(hamming_start_freq * size)
        hamming_window[:start_index] = 1
        filter = ramp_filter * hamming_window
    filter = filter.unsqueeze(-1).unsqueeze(-1)
    sinogram_filtered = []
    for i in range(num_projections):
        img = sinogram_padded[i]
        fft_img = torch.fft.fft(img.T, axis=0)
        projection = fft_img * filter.reshape(-1, 1)
        filtered_projection = torch.fft.ifft(projection, axis=0)[:width, :]
        sinogram_filtered.append(filtered_projection.real.T)
    return torch.stack(sinogram_filtered)


def test_generation():
    n_cells = 1
    print("Generating and organizing yeast cells...")
    all_cells, all_vacuoles, all_lipid_droplets = generate_and_organize(n_cells)
    print("Voxelizing yeast cells...")
    mask, phantom = voxelize_yeast_cells(
        all_cells, all_vacuoles, all_lipid_droplets,
        resolution=(512, 512, 512)
    )
    print("Generating noisy volume...")
    noisy = create_noisy_volume_torch(
        phantom, config=get_default_config()
    )

    print("Reconstructing volume...")
    reconstructed, sino, filtered = reconstruct_3d(
        noisy, theta=100, return_sino=True
    )
    print("Saving center slices...")
    # save center slices
    io.imsave("recon.tif", reconstructed.cpu().numpy())
    io.imsave("noisy.tif", noisy.cpu().numpy())
    io.imsave("phantom.tif", phantom.cpu().numpy())
    io.imsave("projections.tif", torch.swapaxes(sino,0,1).cpu().numpy())
    # normalize the recon and save

    valid_circle = get_valid_voxel_mask(reconstructed, radius_padding=5.0)
    valid_values = reconstructed[valid_circle]
    normed_values = quantile_normalize_tensor(valid_values)
    reconstructed[valid_circle] = normed_values
    reconstructed[~valid_circle] = 0.0

    io.imsave("recon_normed.tif", reconstructed.cpu().numpy())
    print("Done.")

def benchmark(reps: int = 10, n_cells: List[int] = [1, 2, 3]) -> None:
    """
    Benchmark the generation and organization of yeast cells.
    """
    time_rows = []
    # Append header
    time_rows.append(["n_cells","rep", "gen_time", "voxelize_time", "noisy_time", "reconstruct_time", "norm_time"])
    for n in n_cells:
        print(f"Benchmarking {n} cells...")
        for i in range(reps):
            start_timestamp = time.time()
            all_cells, all_vacuoles, all_lipid_droplets = generate_and_organize(n)
            gen_timestamp = time.time()
            mask, phantom = voxelize_yeast_cells(
                all_cells, all_vacuoles, all_lipid_droplets, resolution=(512, 512, 512)
            )
            phantom = torch.from_numpy(phantom).float()
            voxelize_timestamp = time.time()
            noisy = create_noisy_volume_torch(
                phantom, config=get_default_config()
            )
            noisy_timestamp = time.time()
            reconstructed, sino, filtered = reconstruct_3d(
                noisy, theta=100, return_sino=True
            )
            reconstruct_timestamp = time.time()
            valid_circle = get_valid_voxel_mask(reconstructed, radius_padding=5.0)
            valid_values = reconstructed[valid_circle]
            normed_values = quantile_normalize_tensor(valid_values)
            reconstructed[valid_circle] = normed_values
            reconstructed[~valid_circle] = 0.0
            norm_timestamp = time.time()
            time_rows.append([n, i, gen_timestamp - start_timestamp, voxelize_timestamp - gen_timestamp, noisy_timestamp - voxelize_timestamp, reconstruct_timestamp - noisy_timestamp, norm_timestamp - reconstruct_timestamp])

    # Save benchmark results
    df = pd.DataFrame(time_rows[1:], columns=time_rows[0])
    df.to_csv("benchmark_results.csv", index=False)

    # Take mean over repetitions and print
    mean_times = df.groupby("n_cells").mean()
    print(mean_times)


if __name__ == "__main__":
    # Run the test generation
    test_generation()
    # Run the benchmark
    #benchmark(reps=10, n_cells=[1, 2, 3, 4])