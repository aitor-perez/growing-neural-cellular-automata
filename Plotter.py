from matplotlib import pyplot as plt
import utils as ut


class Plotter:

    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.fig = plt.figure()
        self.axs = []

        if batch_size < 2:
            gs = self.fig.add_gridspec(2, 1)
            self.axs.append(self.fig.add_subplot(gs[0, :]))
            self.axs.append(self.fig.add_subplot(gs[1, :]))
        else:
            gs = self.fig.add_gridspec(3, batch_size // 2)

            for k in range(batch_size):
                hor = int(k // (batch_size // 2))
                vert = int(k % (batch_size // 2))
                self.axs.append(self.fig.add_subplot(gs[hor, vert]))

            self.axs.append(self.fig.add_subplot(gs[2, :]))

    def display(self, x, n1, n2):
        images = ut.rechannel_nwhc(ut.to_nwhc(x), 4)

        for k in range(self.batch_size):
            self.axs[k].clear()
            self.axs[k].imshow(images[k].detach().numpy())
            self.axs[k].set_title(str(n1) + " - " + str(n2))
            plt.pause(0.001)

    def plot(self, x):
        self.axs[self.batch_size].clear()
        self.axs[self.batch_size].plot(x)
