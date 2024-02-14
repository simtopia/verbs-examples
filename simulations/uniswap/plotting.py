import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_results(results: List[List[Tuple[int, int]]]):
    n_steps = len(results)
    prices = np.array(results).reshape(n_steps, 2)

    plot_dir = "results/sim_uniswap_gbm"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(prices[:, 0], label="Uniswap price")
    ax.plot(prices[:, 1], label="External market price")
    ax.legend()
    fig.savefig(os.path.join(plot_dir, "prices.pdf"))
