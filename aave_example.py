import argparse

import simulations

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="AAVE agent-based simulation")

    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    parser.add_argument(
        "--n_borrow_agents", type=int, default=10, help="Number of borrowing agents"
    )
    parser.add_argument("--sigma", type=float, default=0.3, help="price volatility")
    parser.add_argument(
        "--n_steps", type=int, default=100, help="Number of steps of the simulation"
    )

    args = parser.parse_args()

    assert (
        0 < args.n_borrow_agents < 100
    ), "Number of borrow agents must be between 0 and 100"

    results = simulations.aave.sim.run_from_cache(
        args.seed, args.n_steps, args.n_borrow_agents, args.sigma
    )

    simulations.aave.plotting.plot_results(results, args.n_borrow_agents)
