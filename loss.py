import torch
import torch.nn.functional as F

def multiclass_dice_loss(pred, target, epsilon=1e-3, ignore_index=-1):
    """
    pred:   (B, C, H, W) logits
    target: (B, 1, H, W) ground truth labels
    """
    num_classes = pred.shape[1]
    pred = torch.softmax(pred, dim=1)
    mask = (target != ignore_index)
    target_masked = target.clone()
    target_masked[~mask] = 0  # Temporarily set ignored pixels to class 0 for one-hot
    target_one_hot = F.one_hot(target_masked, num_classes)
    # Swap dim 1 and -1 (since chan dim should be 1 and empty chan dim can be squeezed)
    target_one_hot = torch.swapaxes(target_one_hot, 1, -1).squeeze(-1).float()
    pred = pred * mask
    target_one_hot = target_one_hot * mask

    # get all but 2nd (channel) dim for summation
    dims = tuple(range(pred.ndim))
    dims = dims[:1] + dims[2:]
    intersection = torch.sum(pred * target_one_hot, dim=dims)
    union = torch.sum(pred + target_one_hot, dim=dims)
    dice_score = (2. * intersection) / (union + epsilon)
    return 1. - (dice_score.mean())


def focal_loss(pred, target, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-1):
    """
    pred:   (B, C, H, W) logits
    target: (B, 1, H, W) ground truth labels
    """
    mask = (target != ignore_index)
    ce_loss = F.cross_entropy(pred, target.squeeze(1), reduction='none', ignore_index=ignore_index)  # (B, H, W)
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    focal = focal * mask  # Zero out ignored pixels
    if reduction == 'mean':
        return focal.sum() / mask.sum().clamp_min(1)
    else:
        return focal.sum()


def dice_focal_loss(
    pred,
    target,
    alpha=1.0,
    gamma=2.0,
    dice_weight=1.0,
    ignore_index=-1,
):
    fl = focal_loss(pred, target, alpha=alpha, gamma=gamma, ignore_index=ignore_index)
    dl = multiclass_dice_loss(pred, target, ignore_index=ignore_index)

    return fl + dice_weight * dl

class DiceFocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=1.0,
        gamma=2.0,
        dice_weight=1.0,
        boundary_weight=0.5,
        ignore_index=-1,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        return dice_focal_loss(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            dice_weight=self.dice_weight,
            ignore_index=self.ignore_index,
        )