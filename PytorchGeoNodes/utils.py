import numpy as np
from networkx.algorithms.bipartite.basic import color
from matplotlib import pyplot as plt

# lower = 0
# upper = 20
# classes = np.arange(lower, upper)
# colormap = plt.cm.jet((classes - lower) / (upper - lower))[..., :3]

# Adapted from:
# https://github.com/Prakadeeswaran05/Semantic-Segmentation-with-DeepLabv3/blob/main/get_dataset_colormap.py
# (Accessed Nov 11 2024)
colormap = np.asarray([
    [0., 0., 0.], # black
    [1., 0., 0.], # red
    [0., 0., 1.], # blue
    [0., 0., 0.5], # blue
    [0., 1., 1.], # cyan
    [0., 1., 0.], # Lime
    [1.0, 0.5, 0], # orange
    [0.5, 0., 1.0], # purple
    [1.0, 0, 0.7], # pink
    [1.0, 1.0, 0], # yellow
    [0.5, 0.8, 0.5], # greenish
    [0.0, 0.5, 0.0], # green
    [0.5, 0.0, 0.0],  # maroon
    [1.0, 0.7, 0.0],  # gold
    # [1.0, 1.0, 1.0],  # white Let's not use white
])