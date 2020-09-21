import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

import utils as ut
from UpdateGrid import UpdateGrid
from Plotter import Plotter

##########################################################################################

# Global variables
N_CHANNELS = 16
BATCH_SIZE = 8
POOL_SIZE = 512

target_name = "macedonIA"

# Hyper-parameters
epochs = 1000
lr_initial = 2e-3
lr_decay_step = 250
lr_decay_factor = 0.1

# Create object containing the neural network
updateGrid = UpdateGrid(N_CHANNELS)

# Criterion is MSE
criterion = nn.MSELoss()

# Optimizer with normalization of gradient
optimizer = optim.Adam(updateGrid.parameters(), lr=lr_initial)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)

for parameter in updateGrid.parameters():
    parameter.register_hook(lambda grad: grad / (torch.norm(grad, 2) + 1e-8))

##########################################################################################

# Open image and convert it to tensor of dimensions (BATCH_SIZE, 4, WIDTH, HEIGHT)
target = ut.read("img/" + target_name + ".png").unsqueeze(0)
target = target.repeat(BATCH_SIZE, 1, 1, 1)

width = target.shape[1]
height = target.shape[2]

# Create object for display
p = Plotter(BATCH_SIZE)

##########################################################################################

# Training
display_step = 10
save_step = 200
losses = []
pool_losses = [0] * POOL_SIZE

# Create initial state with just one alive cell
seed = torch.zeros((1, N_CHANNELS, width, height))
seed[:, 3:, (width - 1) // 2, (height - 1) // 2] = 1

pool = seed.repeat(POOL_SIZE, 1, 1, 1)

for epoch in range(epochs):

    # Pick BATCH_SIZE random samples from the pool
    batch_indices = random.sample(range(POOL_SIZE), BATCH_SIZE)
    batch = pool[batch_indices, :, :, :]

    # Replace the sample with the highest loss with the seed
    batch_losses = [pool_losses[i] for i in batch_indices]
    worst = batch_losses.index(max(batch_losses))
    batch[worst, :, :, :] = seed

    # Feed forward the net a random number of times in [64, 96]
    steps = np.random.randint(64, 97)

    for step in range(steps):
        batch = torch.clamp(updateGrid.forward(batch), 0.0, 1.0)

        if epoch % display_step == 0 and step == steps - 1:
            p.display(batch, epoch, step)

    # Take RGBA channels of the result
    output = ut.rechannel_nwhc(ut.to_nwhc(batch), 4)

    # Empty gradient buffers
    optimizer.zero_grad()

    # Back propagate loss and update parameters
    loss = criterion(output, target)    # Compute loss wrt target
    loss.backward()                     # Back propagate loss to compute differences
    optimizer.step()                    # Update parameters
    scheduler.step()

    # Update pool and pool losses
    pool[batch_indices, :, :, :] = batch
    for i in range(BATCH_SIZE):
        pool_losses[batch_indices[i]] = criterion(output[i], target[i])

    losses.append(loss.detach().numpy())

    # Plot losses
    p.plot(losses)

    # Save model
    if epoch % save_step == save_step - 1 or epoch == epochs - 1:
        model_name = target_name + "-" + str(epochs) + "-" + str(lr_initial) + "-" + str(lr_decay_step) + "-" + str(
            lr_decay_factor)
        torch.save(updateGrid.state_dict(), "models/" + model_name)

    print("[Epoch " + str(epoch) + "] Training L2-loss: ", loss.item(), "; lr: ", str(optimizer.param_groups[0]['lr']))


