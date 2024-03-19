import argparse
import json
import os
from itertools import product

import verbs
from verbs.batch_runner import batch_run

from verbs_examples.aave import plotting, sim
from verbs_examples.utils import post_processing

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="AAVE agent-based simulation")

    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    parser.add_argument(
        "--n_borrow_agents", type=int, default=10, help="Number of borrowing agents"
    )
    parser.add_argument("--sigma", type=float, default=0.3, help="price volatility")
    parser.add_argument("--mu", type=float, default=0.0, help="price drift")
    parser.add_argument(
        "--n_steps", type=int, default=100, help="Number of steps of the simulation"
    )
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

    args = parser.parse_args()

    assert (
        0 < args.n_borrow_agents < 100
    ), "Number of borrow agents must be between 0 and 100"

    with open(os.path.join("verbs_examples", "aave", "cache.json"), "r") as f:
        cache_json = json.load(f)

    if args.cache:
        assert (
            args.alchemy_key is not None
        ), "Alchemy key required, set with '--alchemy_key' argument"
        cache = sim.init_cache(
            args.alchemy_key,
            19163600,
            args.seed,
            args.n_steps,
            args.n_borrow_agents,
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
            n_borrow_agents=args.n_borrow_agents,
            show_progress=False,
        )
        post_processing.save(batch_results, path="results/sim_aave_uniswap")
    else:
        # cache = sim.init_cache(
        #     key="H4UA7VTf-gpUUyhD4GSCDSRvB1Blg3pV",
        #     block_number = 19471508,
        #     seed=args.seed,
        #     n_steps=args.n_steps,
        #     n_borrow_agents=args.n_borrow_agents
        # )

        # run a single simulation
        env = verbs.envs.EmptyEnv(args.seed, cache=cache)

        results = sim.runner(
            env,
            args.seed,
            args.n_steps,
            n_borrow_agents=args.n_borrow_agents,
            mu=args.mu,
            sigma=args.sigma,
            show_progress=True,
        )

        plotting.plot_results(results, args.n_borrow_agents)
