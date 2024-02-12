import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    results: List[List[Tuple]],
    n_borrow_agents: int,
):
    n_steps = len(results)
    records_uniswap_agent = [x[0] for x in results]
    records_borrow_agents = [x[1 : (1 + n_borrow_agents)] for x in results]

    prices = np.array(records_uniswap_agent).reshape(n_steps, 2)
    health_factors = np.array(records_borrow_agents).reshape(n_steps, -1, 2)

    plot_dir = "results/sim_aave_uniswap"

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(prices[:, 0], label="Uniswap price")
    ax.plot(prices[:, 1], label="External market price")
    ax.legend()
    fig.savefig(os.path.join(plot_dir, "prices.pdf"))

    fig, ax = plt.subplots(figsize=(6, 3))

    for i in range(n_borrow_agents):
        hf = health_factors[:, i, :]
        hf = hf[hf[:, 1] < 100, :]
        ax.plot(hf[:, 0], hf[:, 1])

    ax.set_xlabel("simulation step")
    ax.set_ylabel("Health Factor")
    fig.savefig(os.path.join(plot_dir, "health_factors.pdf"))
