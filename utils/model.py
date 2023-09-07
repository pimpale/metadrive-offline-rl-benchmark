import torch
import torch.nn as nn

def deviceof(m: nn.Module) -> torch.device:
    """
    Get the device of the given module
    """
    return next(m.parameters()).device

def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_lr_on_step(step: int, optimizer: torch.optim.Optimizer, lrs: dict[int, float]) -> None:
    """
    Set the learning rate of the optimizer on the given step
    """
    target_lr = optimizer.param_groups[0]['lr']
    for start_step in sorted(lrs.keys()):
        if start_step <= step:
            target_lr = lrs[start_step]
        else:
            break
    if target_lr != optimizer.param_groups[0]['lr']:
        print(f"Set learning rate to {target_lr} on step {step}")
        set_lr(optimizer, target_lr)
