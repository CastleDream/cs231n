################### cosine annealing learning rate schedule #######################
from matplotlib import pyplot
from math import pi
from math import cos
from math import floor


def cosine_annealing(epoch, n_epochs, n_cycles, lrate_max):
    """
    come from https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/
    reference: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    """
    epochs_per_cycle = floor(n_epochs/n_cycles)
    cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
    return lrate_max/2 * (cos(cos_inner) + 1)


# create learning rate series
n_epochs = 100
n_cycles = 5
lrate_max = 0.01
series = [cosine_annealing(i, n_epochs, n_cycles, lrate_max)
          for i in range(n_epochs)]
# plot series
pyplot.plot(series)
pyplot.show()
