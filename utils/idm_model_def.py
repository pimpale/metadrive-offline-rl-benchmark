import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model import deviceof
from utils.transition_model_def import TransitionModel

class InverseDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input shape: (batch_size, 4, 2)
        # output shape: (batch_size, 2)

        self.conv1 = nn.Conv1d(4, 2048, 2) # Bx4x2 -> Bx768x1
        self.fc1 = nn.Linear(2048, 1536) # Bx768 -> Bx768
        self.fc2 = nn.Linear(1536, 1024) # Bx768 -> Bx768
        self.fc3 = nn.Linear(1024, 768) # Bx768 -> Bx768
        self.fc4 = nn.Linear(768, 2) # Bx768 -> Bx2
    
    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.clamp(x, -1, 1)
        return x

def idm_train_batch(
        tm: TransitionModel,
        idm: InverseDynamicsModel,
        idm_optimizer: torch.optim.Optimizer,
        obs_tensor: torch.Tensor,
        s0_tensor: torch.Tensor,
        s1_tensor: torch.Tensor,
) -> float:
    device = deviceof(tm)
    assert deviceof(idm) == device

    obs_tensor = obs_tensor.to(device)
    s0_tensor = s0_tensor.to(device)
    s1_tensor = s1_tensor.to(device)

    idm_optimizer.zero_grad()

    pred_action = idm(obs_tensor)
    pred_s1 = tm(s0_tensor, pred_action)

    loss = F.mse_loss(pred_s1, s1_tensor)
    loss.backward()

    idm_optimizer.step()

    return float(loss.item())