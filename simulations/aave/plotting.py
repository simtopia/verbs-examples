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
    records_liquidator = [x[(1 + n_borrow_agents) :] for x in results]

    prices = np.array(records_uniswap_agent).reshape(n_steps, 2)
    results = np.array(records_borrow_agents).reshape(n_steps, -1, 4)
    records_liquidator = np.array(records_liquidator).reshape(n_steps, -1)

    plot_dir = "results/sim_aave_uniswap"

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(prices[:, 0], label="Uniswap price")
    ax.plot(prices[:, 1], label="External market price")
    ax.legend()
    fig.savefig(os.path.join(plot_dir, "prices.pdf"))

    fig, ax = plt.subplots(figsize=(6, 9), nrows=3)

    for i in range(n_borrow_agents):
        res = results[:, i, :]
        res = res[res[:, 1] < 100, :]
        ax[0].plot(res[:, 0], res[:, 1], label=f"Borrower {i}")
        ax[1].plot(res[:, 0], res[:, 2] / 10**8, label=f"Borrower {i}")
        ax[2].plot(res[:, 0], res[:, 3] / 10**8, label=f"Borrower {i}")

    ax[2].set_xlabel("simulation step")
    ax[0].set_ylabel("Health Factor")
    ax[1].set_ylabel("Collateral base")
    ax[2].set_ylabel("Debt base")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "borrowers.pdf"))

    fig, ax = plt.subplots(figsize=(6, 6), nrows=2)
    ax[0].plot(records_liquidator[:, 0] / 10**18)
    ax[0].set_title("Balance collateral asset")
    ax[1].plot(records_liquidator[:, 1] / 10**18)
    ax[1].set_title("Balance debt asset")
    fig.savefig(os.path.join(plot_dir, "liquidator.pdf"))
