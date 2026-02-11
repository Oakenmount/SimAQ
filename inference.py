import numpy as np
import torch
from typing import Optional, List
import argparse

from tifffile import imwrite, imread
from omegaconf import OmegaConf, ListConfig, DictConfig
from tqdm import tqdm

from unet3d2d import SimAQModel
from datagenerator import quantile_normalize_tensor

def parse_args():
    parser = argparse.ArgumentParser(description="Inference on 3D images")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--input', type=str, required=True, help='Path to input image (TIFF)')
    parser.add_argument('--output', type=str, default='prediction.tif', help='Path to output file')
    parser.add_argument('--overrides', nargs='*', default=[], help='Config overrides')
    return parser.parse_args()

def load_config(config_path: str, overrides: Optional[List] = None) -> ListConfig | DictConfig:
    """Load and merge configuration."""
    cfg = OmegaConf.load(config_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    OmegaConf.resolve(cfg)
    return cfg

def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load input image
    img = imread(args.input).astype(np.float32)

    # Model
    model = SimAQModel(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        init_features=cfg.model.feature_size,
    ).to(device)

    # Load weights
    print(f"Loading model weights from {args.weights}")
    weights = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(weights, strict=True)

    in_tensor = torch.from_numpy(img).float()
    in_tensor = quantile_normalize_tensor(in_tensor)
    out_tensor = torch.zeros_like(in_tensor)
    padding = cfg.data.padding
    # Inference
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(padding, in_tensor.shape[0] - padding)):
            img_slice = in_tensor[i - padding:i + padding + 1, :, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
            img_slice = img_slice.to(device)
            pred_slice = model(img_slice)
            out_tensor[i, :, :] = torch.argmax(pred_slice, dim=1).squeeze().cpu()

    # Save prediction
    imwrite(args.output, out_tensor.squeeze(1).cpu().numpy().astype(np.uint8))
    print(f"Prediction saved to {args.output}")


if __name__ == "__main__":
    main()
