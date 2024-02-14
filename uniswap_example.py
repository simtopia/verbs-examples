import argparse

import simulations

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Uniswap agent-based simulation")

    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    parser.add_argument(
        "--n_steps", type=int, default=100, help="Number of steps of the simulation"
    )
    args = parser.parse_args()

    results = simulations.uniswap.sim.run_from_cache(args.seed, args.n_steps)

    simulations.uniswap.plotting.plot_results(results)
