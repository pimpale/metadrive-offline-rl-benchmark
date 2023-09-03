import torch
import torch.nn as nn
import torch.nn.functional as F

from env import State, Action, state_batch_to_tensor, action_batch_to_tensor
from model_utils import deviceof

# create a model that attempts to predict the next state given the current state and the action: (throttle and steering)
# each state contains: velocity_x, velocity_y, and heading
class MetadriveModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input shape: (batch_size, 3) + (batch_size, 2) = (batch_size, 5)
        # output shape: (batch_size, 3)
        self.fc1 = nn.Linear(6, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 768)
        self.fc4 = nn.Linear(768, 768)
        self.fc5 = nn.Linear(768, 4)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        # clip actions to be between -1 and 1
        actions = torch.clamp(actions, -1, 1)
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x) # no activation function on the last layer
        x = states + x
        return x

def metadrive_model_train_batch(
    mm: MetadriveModel,
    mm_optimizer: torch.optim.Optimizer,
    s0_tensor: torch.Tensor,
    a_tensor: torch.Tensor,
    s1_tensor: torch.Tensor
) -> float: 
    device = deviceof(mm)
    
    s0_tensor = s0_tensor.to(device)
    a_tensor = a_tensor.to(device)
    s1_tensor = s1_tensor.to(device)

    mm_optimizer.zero_grad()
    s1_pred_tensor = mm(s0_tensor, a_tensor)
    loss = F.mse_loss(s1_pred_tensor, s1_tensor)
    loss.backward()
    mm_optimizer.step()
    return float(loss.item())