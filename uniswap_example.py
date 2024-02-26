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
    parser.add_argument("--sigma", type=float, default=0.3, help="GBM volatility")
    parser.add_argument(
        "--batch_runner",
        action="store_true",
        help="Run batch of simulations over different simulation parameters",
    )
    args = parser.parse_args()

    with open(os.path.join("simulations", "uniswap", "cache.json"), "r") as f:
        cache_json = json.load(f)

    cache = verbs.utils.cache_from_json(cache_json)

    if args.batch_runner:
        # run a batch of simulations
        parameters_samples = [
            dict(mu=mu, sigma=sigma)
            for mu, sigma in product([0.0, 0.1, -0.1], [0.1, 0.2, 0.3])
        ]

        batch_results = batch_run(
            simulations.uniswap.sim.runner,
            n_steps=args.n_steps,
            n_samples=10,
            parameters_samples=parameters_samples,
            cache=cache,
            show_progress=False,
        )
        simulations.utils.post_processing.save(
            batch_results, path="results/sim_uniswap_gbm"
        )
    else:
        # single simulation
        env = verbs.envs.EmptyEnv(args.seed, cache=cache)

        results = simulations.uniswap.sim.runner(
            env,
            args.seed,
            args.n_steps,
            mu=0.0,
            sigma=args.sigma,
            show_progress=True,
        )
        simulations.uniswap.plotting.plot_results(results)
