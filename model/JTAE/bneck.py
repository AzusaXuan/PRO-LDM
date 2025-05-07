from torch import nn

class BaseBottleneck(nn.Module):

    def __init__(self, input_dim, bottleneck_dim):
        super(BaseBottleneck, self).__init__()

        self.fc1 = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, h):

        z_rep = self.fc1(h)

        return z_rep


