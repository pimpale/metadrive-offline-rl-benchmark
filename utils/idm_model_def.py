import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model import deviceof

class InverseDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input shape: (batch_size, 4, 2)
        # output shape: (batch_size, 2)

        self.conv1 = nn.Conv1d(4, 2048, 2)
        self.fc1 = nn.Linear(2048, 1536)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1536, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 768)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(768, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(512, 2)
    

    # performs the forward pass on the ego-frame transformed input
    def forward_ego_frame(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.fc5(x)
        x = torch.clamp(x, -1, 1)
        return x

    def forward(self, x: torch.Tensor):
        # rotate the vectors from the world frame to the ego frame before passing them through the network
        # note: since we're rotating from the world frame to the ego frame, we need to use the negative of the heading
        c = x[:, 2, 0]
        s = -x[:, 3, 0]
        rot = torch.stack([
                torch.stack([c, -s]),
                torch.stack([s, c])
        ]).permute(2, 0, 1)

        v0 = x[:, 0:2, 0]
        d0 = x[:, 2:4, 0]
        v1 = x[:, 0:2, 1]
        d1 = x[:, 2:4, 1]

        v0_rot = torch.bmm(rot, v0.unsqueeze(2)).squeeze(2)
        d0_rot = torch.bmm(rot, d0.unsqueeze(2)).squeeze(2)
        v1_rot = torch.bmm(rot, v1.unsqueeze(2)).squeeze(2)
        d1_rot = torch.bmm(rot, d1.unsqueeze(2)).squeeze(2)

        x_ego_frame = torch.stack([torch.cat([v0_rot, d0_rot], 1), torch.cat([v1_rot, d1_rot], 1)], 2)
        return self.forward_ego_frame(x_ego_frame)

def idm_train_direct_batch(
        idm: InverseDynamicsModel,
        idm_optimizer: torch.optim.Optimizer,
        obs_tensor: torch.Tensor,
        a_tensor: torch.Tensor,
) -> float:
    device = deviceof(idm)

    obs_tensor = obs_tensor.to(device)
    a_tensor = a_tensor.to(device)

    idm_optimizer.zero_grad()

    pred_action = idm(obs_tensor)

    loss = F.mse_loss(pred_action, a_tensor)
    loss.backward()

    idm_optimizer.step()

    return float(loss.item())