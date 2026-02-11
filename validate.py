import numpy as np
import torch
from typing import Optional, List
import argparse

from tifffile import imwrite
from omegaconf import OmegaConf, ListConfig, DictConfig

from unet3d2d import SimAQModel
from datagenerator import ValidationSemi3DDataGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="Test the model with 3D data")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights')
    parser.add_argument('--overrides', nargs='*', default=[], help='Config overrides (e.g., training.batch_size=8)')
    parser.add_argument('--theta', type=int, default=None, help='Theta angle for validation')
    return parser.parse_args()

def load_config(config_path: str, overrides: Optional[List] = None) -> ListConfig | DictConfig:
    """Load and merge configuration."""
    # Load base config
    cfg = OmegaConf.load(config_path)
    
    # Apply CLI overrides if provided
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    
    # Resolve interpolations
    OmegaConf.resolve(cfg)
    
    return cfg

def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    if args.theta is not None:
        theta_distribution = np.array([int(args.theta)])
    else:
        theta_distribution = np.arange(
            cfg.data.theta_range.start,
            cfg.data.theta_range.end,
            cfg.data.theta_range.step
        )
    

    dataset = ValidationSemi3DDataGenerator(
        thetas=theta_distribution, # type: ignore
        size=cfg.data.resolution,
        padding=cfg.data.padding,
    )

    # Model
    model = SimAQModel(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        init_features=cfg.model.init_features,
        depth=cfg.model.depth,
        skip_first=cfg.model.skip_first,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    ).to(device)

    # Load weights if provided
    if args.weights: # easy override from CLI
        cfg.training.weights_path = args.weights
    if cfg.training.weights_path:
        try:
            print(f"Loading model weights from {cfg.training.weights_path}")
            weights = torch.load(cfg.training.weights_path, map_location='cpu')
            model.load_state_dict(weights, strict=True)
        except RuntimeError as e:
            print("Mismatch in model and weights. Loading those that match.")
            current_model_dict = model.state_dict()
            weights = torch.load(cfg.training.weights_path, map_location='cpu')
            new_state_dict = {
                k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] 
                for k, v in zip(current_model_dict.keys(), weights.values())
            }
            model.load_state_dict(new_state_dict, strict=False)


    noisy, mask = next(iter(dataset))
    model.eval()
    with torch.no_grad():
        pred = model(noisy.unsqueeze(0).float().to(device))
    pred = torch.argmax(pred, dim=1)
    pred = pred.squeeze().cpu().numpy()

    # Save noisy, mask, and prediction as TIFF files
    imwrite('noisy.tif', noisy.squeeze().cpu().numpy().astype(np.float32))
    imwrite('mask.tif', mask.squeeze().cpu().numpy().astype(np.uint8))
    imwrite('prediction.tif', pred.astype(np.uint8))


if __name__ == "__main__":
    main()