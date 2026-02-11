import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, List
import argparse

import lightning as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins import TorchSyncBatchNorm

from omegaconf import OmegaConf, ListConfig, DictConfig

from unet3d2d import SimAQModel
from datagenerator import Semi3DDataGenerator, RealSemi3DDataGenerator, MixedDataGenerator, ValidationSemi3DDataGenerator
from augmentations import Augmentor
from callbacks import UpdateEMACallback, SaveWeights, VisualLoggingCallback, ClearCache
from util import EMAWrapper
from loss import DiceFocalLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Train model with 3D data")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights')
    parser.add_argument('--overrides', nargs='*', default=[], help='Config overrides (e.g., training.batch_size=8)')
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

    # Initialize WandB logger
    logger = WandbLogger(
        project=cfg.wandb.project,
        save_dir=cfg.wandb.save_dir,
        log_model=cfg.wandb.log_model,
        config=OmegaConf.to_container(cfg)
    )

    # Data
    theta_distribution = np.arange(
        cfg.data.theta_range.start,
        cfg.data.theta_range.end,
        cfg.data.theta_range.step
    )
    
    synth_generator = Semi3DDataGenerator(
            thetas=theta_distribution,
            buffer_max_capacity=cfg.data.buffer_max_capacity,
            size=cfg.data.resolution,
            padding=cfg.data.padding,
            num_cycles=cfg.data.num_cycles,
            num_steps=cfg.training.epoch_steps,
            include_phantom=False,
            background_filter_rate=0.98 # Most of the synthetic data is empty
        )

    if cfg.training.real_ratio > 0.0:
        real_generator = RealSemi3DDataGenerator(
            data_path=cfg.data.real_data_dir,
            resolution=cfg.data.resolution,
            padding=cfg.data.padding
        )
        mixed_datagen = MixedDataGenerator(
            synthetic_generator=synth_generator,
            real_data_generator=real_generator,
            real_ratio=cfg.training.real_ratio,
            num_steps=cfg.training.epoch_steps
        )
        datamodule = DataLoader(
            dataset=mixed_datagen, 
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers
        )
    else:
        datamodule = DataLoader(
            dataset=synth_generator, 
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers
        )

    augmentor = None
    if cfg.augmentations.enable:
        augmentor = Augmentor(cfg)

    # Criterion
    criterion = DiceFocalLoss(
        alpha=cfg.loss.alpha,
        gamma=cfg.loss.gamma,
        dice_weight=cfg.loss.dice_weight,
        ignore_index=cfg.loss.ignore_index
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
        augmentations=augmentor,
        criterion=criterion
    )

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

    # Make checkpoint directory if it doesn't exist
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_name = str(logger.experiment.name)
    run_name = run_name.replace("-", "_")

    # Callbacks
    callbacks = []

    # Save weights
    if cfg.training.ema_enabled:
        ema_model = EMAWrapper(model, decay=cfg.ema.decay)
        callbacks.append(UpdateEMACallback(ema_model=ema_model))
        callbacks.append(SaveWeights(
            log_dir=cfg.training.checkpoint_dir, 
            run_name=run_name, 
            ema_model=ema_model
        ))
    else:
        callbacks.append(SaveWeights(
            log_dir=cfg.training.checkpoint_dir, 
            run_name=run_name
        ))
        ema_model = None

    # Visual samples
    val_generator = ValidationSemi3DDataGenerator(
        thetas=theta_distribution,
        size=cfg.data.resolution,
        padding=cfg.data.padding,
    )

    callbacks.append(VisualLoggingCallback(
        validation_generator=val_generator,
        log_interval=1, # Basically free
        ema_model=ema_model
    ))

    # Clear cache callback
    callbacks.append(ClearCache())

    # Enable mixed precision training if available
    torch.set_float32_matmul_precision(cfg.torch.float32_matmul_precision)
    
    # Sync batch normalization
    if cfg.training.sync_batchnorm:
        model = TorchSyncBatchNorm().apply(model)

    # Create trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.training.epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=DDPStrategy() if torch.cuda.device_count() > 1 else 'auto',
        sync_batchnorm=cfg.training.sync_batchnorm,
        precision=cfg.training.precision if torch.cuda.is_available() else 32,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        enable_checkpointing=cfg.trainer.enable_checkpointing
    )

    # Training
    trainer.fit(model, datamodule) # type: ignore

if __name__ == "__main__":
    main()