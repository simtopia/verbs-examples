import argparse
import json
import os
from itertools import product

import verbs
from verbs.batch_runner import batch_run

import simulations

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Uniswap agent-based simulation")

    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    parser.add_argument(
        "--n_steps", type=int, default=100, help="Number of steps of the simulation"
    )
    parser.add_argument(
        "--batch_runner",
        action="store_true",
        help="Run batch of simulations over different simulation parameters",
    )
    args = parser.parse_args()

    # run a single simulation
    results = simulations.uniswap.sim.run_from_cache(args.seed, args.n_steps)
    simulations.uniswap.plotting.plot_results(results)

    # run a batch of simulations
    if args.batch_runner:
        parameters_samples = [
            dict(mu=mu, sigma=sigma)
            for mu, sigma in product([0.0, 0.1, -0.1], [0.1, 0.2, 0.3])
        ]

        with open(os.path.join("simulations", "uniswap", "cache.json"), "r") as f:
            cache_json = json.load(f)
        cache = verbs.utils.cache_from_json(cache_json)

        batch_results = batch_run(
            simulations.uniswap.sim.runner,
            n_steps=args.n_steps,
            n_samples=10,
            parameters_samples=parameters_samples,
            cache=cache,
        )
        simulations.utils.postprocessing.save(
            batch_results, path="simulations/results/sim_uniswap_gbm"
        )
