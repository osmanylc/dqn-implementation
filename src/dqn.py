from torch import nn
from torch.nn import functional as F


class QNet(nn.Module):
    
    def __init__(self, num_actions):
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 
                               kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 
                               kernel_size=4, stride=2)
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, num_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((x.shape[0], -1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
