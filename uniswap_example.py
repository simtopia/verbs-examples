import argparse
import json
import os
from itertools import product

import verbs
from verbs.batch_runner import batch_run

from verbs_examples.uniswap import plotting, sim
from verbs_examples.utils import post_processing

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Uniswap agent-based simulation")

    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    parser.add_argument(
        "--n_steps", type=int, default=100, help="Number of steps of the simulation"
    )
    parser.add_argument("--sigma", type=float, default=0.3, help="GBM volatility")
    parser.add_argument("--mu", type=float, default=0.0, help="GBM drift")
    parser.add_argument(
        "--batch_runner",
        action="store_true",
        help="Run batch of simulations over different simulation parameters",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Generate a new request cache file.",
    )
    parser.add_argument(
        "--alchemy_key",
        type=str,
        help="Generate a new request cache file.",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=19163600,
        help="Ethereum Block number for mainnet forking",
    )
    args = parser.parse_args()

    with open(os.path.join("verbs_examples", "uniswap", "cache.json"), "r") as f:
        cache_json = json.load(f)

    if args.cache:
        assert (
            args.alchemy_key is not None
        ), "Alchemy key required, set with '--alchemy_key' argument"
        cache = sim.init_cache(
            args.alchemy_key,
            args.block,
            args.seed,
            args.n_steps,
        )
    else:
        cache = verbs.utils.cache_from_json(cache_json)

    if args.batch_runner:
        # run a batch of simulations
        parameters_samples = [
            dict(mu=mu, sigma=sigma)
            for mu, sigma in product([0.0, 0.1, -0.1], [0.1, 0.2, 0.3])
        ]

        batch_results = batch_run(
            sim.runner,
            n_steps=args.n_steps,
            n_samples=10,
            parameters_samples=parameters_samples,
            cache=cache,
            show_progress=False,
        )
        post_processing.save(batch_results, path="results/sim_uniswap_gbm")
    else:
        # single simulation
        env = verbs.envs.EmptyEnvRandom(args.seed, cache=cache)

        results = sim.runner(
            env,
            args.seed,
            args.n_steps,
            mu=args.mu,
            sigma=args.sigma,
            show_progress=True,
        )
        plotting.plot_results(results)
