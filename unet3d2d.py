import torch
import torch.nn as nn

import lightning as pl

from typing import Optional

from loss import DiceFocalLoss

class CenterSliceUNet3D2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64, depth=4, skip_first=True):
        super().__init__()

        if depth < 2:
            raise ValueError("depth must be >= 2")

        F = init_features

        self.depth = depth
        self.skip_first = skip_first

        # -------- 3D Encoder --------
        self.encoders = nn.ModuleList()
        for i in range(depth):
            in_ch = in_channels if i == 0 else F * (2 ** (i - 1))
            out_ch = F * (2 ** i)
            self.encoders.append(self._block_3d(in_ch, out_ch))

        # 2D pooling (applied slice-wise)
        self.pool2d = nn.MaxPool2d(2)

        # -------- 2D Decoder --------
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in range(depth - 1, 0, -1):
            in_ch = F * (2 ** i)
            out_ch = F * (2 ** (i - 1))
            self.upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))

        for i in range(depth - 1):
            out_ch = F * (2 ** i)
            if i == 0 and skip_first:
                dec_in_ch = out_ch
            else:
                dec_in_ch = out_ch * 2
            self.dec_blocks.append(self._block_2d(dec_in_ch, out_ch))

        self.final_conv = nn.Conv2d(F, out_channels, kernel_size=1)

    # -------- blocks --------
    def _block_3d(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
        )

    def _block_2d(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    # -------- forward --------
    def forward(self, x):
        # x: [B, C, D, H, W]
        center_idx = x.shape[2] // 2

        # ---- Encoder ----
        enc_outputs = []
        x_enc = x
        for i, enc in enumerate(self.encoders):
            x_enc = enc(x_enc)
            enc_outputs.append(x_enc)
            if i < self.depth - 1:
                x_enc = self._apply_2d_pool(x_enc)

        # ---- Center slice bridge ----
        d = enc_outputs[-1][:, :, center_idx]

        # ---- Decoder ----
        for level in range(self.depth - 2, -1, -1):
            up_idx = (self.depth - 2) - level
            d = self.upconvs[up_idx](d)

            if not (self.skip_first and level == 0):
                skip = enc_outputs[level][:, :, center_idx]
                d = torch.cat([d, skip], dim=1)

            d = self.dec_blocks[level](d)

        return self.final_conv(d)

    # -------- utils --------
    def _apply_2d_pool(self, x):
        """
        Slice-wise 2D pooling.
        x: [B, C, D, H, W] → [B, C, D, H/2, W/2]
        """
        B, C, D, H, W = x.shape
        x = x.view(B * D, C, H, W)
        x = self.pool2d(x)
        _, _, H2, W2 = x.shape
        return x.view(B, C, D, H2, W2)


class SimAQModel(pl.LightningModule):
    def __init__(self, in_channels=1, 
                 out_channels=1,
                 init_features=64,
                 depth=4,
                 skip_first=True,
                 lr=1e-4, 
                 weight_decay=1e-5, 
                 augmentations: Optional[torch.nn.Module]=None, 
                 criterion: Optional[torch.nn.Module]=None):
        super(SimAQModel, self).__init__()
        self.unet3d2d = CenterSliceUNet3D2D(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            depth=depth,
            skip_first=skip_first,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.augmentations = augmentations
        self.criterion = criterion if criterion is not None else DiceFocalLoss(alpha=1.0, gamma=2.0, dice_weight=1.0, ignore_index=-1)
    
    def forward(self, x):
        return self.unet3d2d(x)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        noisy, mask = batch
        if self.augmentations:
            noisy, mask = self.augmentations(noisy, mask)
        outputs = self(noisy)
        loss = self.criterion(outputs, mask)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer with specific parameter exclusions"""
        
        # Collect different types of parameters
        prelu_params = []
        bias_params = []
        batchnorm_params = []
        regular_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'prelu' in name.lower() and 'weight' in name.lower():
                prelu_params.append(param)
            elif 'bias' in name.lower():
                bias_params.append(param)
            elif any(bn in name.lower() for bn in ['batchnorm', 'bn', 'running_mean', 'running_var']):
                batchnorm_params.append(param)
            else:
                regular_params.append(param)
        
        # Create optimizer with multiple parameter groups
        optimizer = torch.optim.Adam([
            {
                'params': regular_params,
                'lr': self.lr,
                'weight_decay': self.weight_decay
            },
            {
                'params': prelu_params,
                'lr': self.lr,
                'weight_decay': 0.0
            },
            {
                'params': bias_params,
                'lr': self.lr,
                'weight_decay': 0.0
            },
            {
                'params': batchnorm_params,
                'lr': self.lr,
                'weight_decay': 0.0
            }
        ])
        
        return optimizer


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("=" * 60)
    print("Testing CenterSliceUNet3D2D Model")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("\n1. Creating model...")
    model = CenterSliceUNet3D2D(in_channels=1, init_features=64, depth=5, skip_first=True)
    model = model.to(device)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"   Model created successfully!")
    print(f"   Number of parameters: {num_params:,}")
    
    # Create test input tensor
    print("\n2. Creating test input tensor...")
    batch_size = 8
    channels = 1
    depth = 7
    height = 512
    width = 512
    
    # Create empty tensor (zeros)
    input_tensor = torch.zeros(batch_size, channels, depth, height, width)
    input_tensor = input_tensor.to(device)
    
    print(f"   Input tensor shape: {input_tensor.shape}")
    print(f"   Input tensor device: {input_tensor.device}")
    print(f"   Input tensor dtype: {input_tensor.dtype}")
    
    # Run forward pass
    print("\n3. Running forward pass...")
    try:
        model.eval()  # Set to evaluation mode
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"   Forward pass successful!")
        print(f"   Output tensor shape: {output.shape}")
        print(f"   Output tensor device: {output.device}")
        print(f"   Output tensor dtype: {output.dtype}")
        
        # Verify output shape
        expected_shape = (batch_size, 1, height, width)
        if output.shape == expected_shape:
            print(f"   ✓ Output shape matches expected: {expected_shape}")
        else:
            print(f"   ✗ Output shape mismatch! Expected: {expected_shape}, Got: {output.shape}")
        
        # Check output range (should be reasonable for uninitialized model)
        output_min = output.min().item()
        output_max = output.max().item()
        output_mean = output.mean().item()
        print(f"   Output stats - Min: {output_min:.6f}, Max: {output_max:.6f}, Mean: {output_mean:.6f}")
        
    except Exception as e:
        print(f"   ✗ Forward pass failed with error: {e}")
        raise
    
    # Test memory usage (if CUDA is available)
    if torch.cuda.is_available():
        print("\n5. GPU Memory Usage:")
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
        print(f"   Peak GPU memory usage: {peak_memory:.2f} GB")
        
        # Clean up
        del input_tensor, output
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("Model verification complete! ✓")
    print("=" * 60)
    
    # Optional: Save model architecture summary
    print("\nModel Architecture Summary:")
    print("-" * 40)
    print(f"Input:  [B, {channels}, {depth}, {height}, {width}]")
    print(f"Output: [B, 1, {height}, {width}]")
    print(f"Parameters: {num_params:,}")
    print(f"Device: {device}")