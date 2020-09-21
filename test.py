import torch

from UpdateGrid import UpdateGrid
from Plotter import Plotter
from matplotlib import pyplot as plt

N_CHANNELS = 16
width = 32
height = 32

path = "models/" + "basic-1000-0.002-500-0.1"

updateGrid = UpdateGrid(N_CHANNELS)
updateGrid.load_state_dict(torch.load(path))

seed = torch.zeros((1, N_CHANNELS, width, height))
seed[:, 3:, (width - 1) // 2, (height - 1) // 2] = 1

p = Plotter(1)

steps = 64 + int(32*torch.rand(1))
for step in range(steps):
    seed = torch.clamp(updateGrid.forward(seed), 0.0, 1.0)
    p.display(seed, 0, step)
    plt.pause(0.01)

plt.pause(10)

