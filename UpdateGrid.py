import torch
import torch.nn as nn
import torch.nn.functional as F


class UpdateGrid(nn.Module):

    def __init__(self, n_channels):
        super(UpdateGrid, self).__init__()

        self.n_channels = n_channels

        # Filters: sobel_x, sobel_y and identity
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).repeat(n_channels, 1, 1, 1).type(torch.float)
        sobel_y = torch.transpose(sobel_x, 2, 3)
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).repeat(n_channels, 1, 1, 1).type(torch.float)
        self.filters = [sobel_x, sobel_y, identity]

        self.fc1 = nn.Conv2d(n_channels * len(self.filters), 128, 1)
        self.fc2 = nn.Conv2d(128, n_channels, 1)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        batch_size = x.shape[0]
        width = x.shape[2]
        height = x.shape[3]

        # Preparation of perception vector as concatenation of convolutions with the filters above
        perception = torch.zeros((batch_size, len(self.filters) * self.n_channels, width, height))

        for i, filt in enumerate(self.filters):
            perception_chunk = F.conv2d(x, filt, groups=self.n_channels, padding=1)
            perception[:, (i * self.n_channels):((i+1) * self.n_channels), :, :] = perception_chunk

        # Architecture of the neural network
        dx = self.fc1(perception)
        dx = F.relu(dx)
        dx = self.fc2(dx)

        # Stochastic update
        random_mask = torch.rand(batch_size, 1, width, height)
        random_mask = random_mask.repeat(1, self.n_channels, 1, 1)       # adapt dimensions (check if needed at all)

        x = x + dx * random_mask

        # Alive masking
        mature = (x[:, 3:4, :, :] > 0.1).type(torch.int)  # extract alive cells

        alive_kernel = torch.ones(1, 1, 3, 3).type(torch.int)  # prepare convolution kernel to check all neighbors

        alive = F.conv2d(mature, alive_kernel, padding=1)  # count mature neighbors. If any, cell is alive
        alive = (alive > 0).type(torch.int)                # normalize
        alive = alive.repeat(1, self.n_channels, 1, 1)          # adapt dimensions (check if needed at all)

        return x * alive

