from typing import Optional

import torch
from util import EMAWrapper
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
from datagenerator import SyntheticGenerator


class SaveWeights(pl.Callback):
    def __init__(self, log_dir: str, run_name: str, ema_model: Optional[torch.nn.Module] = None):
        super().__init__()
        self.log_dir = log_dir
        self.run_name = run_name
        self.ema_model = ema_model

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        # Save the model weights
        save_path = f"{self.log_dir}/{self.run_name}_weights.pth"
        torch.save(pl_module.state_dict(), save_path)

        # If using EMA, save the EMA model weights
        if isinstance(self.ema_model, EMAWrapper):
            ema_save_path = f"{self.log_dir}/{self.run_name}_ema_weights.pth"
            torch.save(self.ema_model.module.state_dict(), ema_save_path)


class UpdateEMACallback(pl.Callback):
    def __init__(self, ema_model: EMAWrapper):
        super().__init__()
        self.ema_model = ema_model

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx):
        # Only update EMA if on primary process
        if not trainer.is_global_zero:
            return

        # Make sure it is on the same device as the model
        if self.ema_model.device != pl_module.device:
            self.ema_model.to(pl_module.device)

        # Update the EMA model parameters
        self.ema_model.update_parameters(pl_module)

class ClearCache(pl.Callback):
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module):
        # Clear cache to free up memory
        torch.cuda.empty_cache()

class VisualLoggingCallback(pl.Callback):
    def __init__(self, validation_generator, log_interval: int = 100, ema_model: Optional[EMAWrapper]=None):
        super().__init__()
        self.validation_generator = validation_generator
        self.log_interval = log_interval
        self.ema_model = ema_model

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Only log on primary process
        if not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch
        if epoch % self.log_interval == 0:
            noisy, mask = next(iter(self.validation_generator))

            noisy = noisy.to(pl_module.device, dtype=torch.half)
            # AMP and no grad for validation
            with torch.amp.autocast("cuda"), torch.no_grad(): # type: ignore
                if self.ema_model is not None:
                    self.ema_model.to(pl_module.device)
                    prediction = self.ema_model.module(noisy.unsqueeze(0)).squeeze(0)
                    self.ema_model.to('cpu')
                else:
                    prediction = pl_module(noisy.unsqueeze(0)).squeeze(0)

            center_slice = noisy.shape[1] // 2
            noisy = noisy.cpu().float().squeeze(0)

            prediction = torch.argmax(prediction, dim=0).cpu().to(torch.uint8)

            mask = mask.cpu().to(torch.uint8).squeeze(0)

            # log noisy center slice and the prediction and ground truth masks
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(noisy[center_slice], cmap='gray')
            axs[0].set_title('Noisy Input (Center Slice)')
            axs[0].axis('off')
            axs[1].imshow(prediction, cmap='jet', vmin=0, vmax=prediction.max())
            axs[1].set_title('Predicted Mask (Center Slice)')
            axs[1].axis('off')
            axs[2].imshow(mask, cmap='jet', vmin=0, vmax=mask.max())
            axs[2].set_title('Ground Truth Mask (Center Slice)')
            axs[2].axis('off')
            plt.tight_layout()
            assert isinstance(trainer.logger, WandbLogger)
            trainer.logger.experiment.log({f'validation_samples': plt})
            plt.close(fig)
