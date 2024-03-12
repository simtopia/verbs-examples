"""
Uniswap simulation implementation

In this example we model an agent that trades between a Uniswap pool and
and an external market, modelled by a Geometric Brownian Motion, in order
to make a profit.

* We consider the Uniswap v3 pool for WETH and DAI with fee 3000.
* The price of the risky asset (WETH) in terms of the stablecoin (DAI) in the
  external market is modelled by a GBM.

The goal of the simulation is for the price of Uniswap to follow the price
in the external market. The Uniswap agent takes of that in each step, by
making the right trade so that the new Uniswap price is the same as the
price in the external market.

"""

import json
from functools import partial
from pathlib import Path
from typing import List

import verbs

from simulations import abis
from simulations.agents import DummyUniswapAgent, UniswapAgent
from simulations.utils.erc20 import mint_and_approve_dai, mint_and_approve_weth

PATH = Path(__file__).parent

WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
DAI_ADMIN = "0x9759A6Ac90977b93B58547b4A71c78317f391A28"
UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
# sanity check, obtained from the factory contract using web3.py
UNISWAP_WETH_DAI = "0xC2e9F25Be6257c210d7Adf0D4Cd6E3E881ba25f8"
SWAP_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
UNISWAP_QUOTER = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"


def runner(
    env,
    seed: int,
    n_steps: int,
    *,
    mu: float = 0.0,
    sigma: float = 0.3,
    uniswap_agent_type=UniswapAgent,
    show_progress=True,
) -> List[List]:
    """
    Uniswap simulation runner

    Parameters
    ----------
    env
        Simulation environment
    seed: int
        Random seed
    n_steps: int
        Number of simulation steps
    mu: float, optional
        GBM mu parameter, default 0.0
    sigma: float, optional
        GBM sigma parameter, default 0.3
    uniswap_agent_type: BaseUniswapAgent, optional
        Either UniswapAgent or DummyUniswapAgent, depending
        on wheter the simulation is initialising the Cache or not
    show_progress: bool, optional
        If ``True`` simulation progress will be printed

    Returns
    -------
    list
        List of agent states recorded over the simulation
    """

    # Convert addresses
    weth_address = verbs.utils.hex_to_bytes(WETH)
    dai_address = verbs.utils.hex_to_bytes(DAI)
    swap_router_address = verbs.utils.hex_to_bytes(SWAP_ROUTER)
    quoter_address = verbs.utils.hex_to_bytes(UNISWAP_QUOTER)
    dai_admin_address = verbs.utils.hex_to_bytes(DAI_ADMIN)

    # Example: Use uniswap_factory contract to get the address of WETH-DAI
    # pool with fee 3000
    fee = 3000

    pool_address = abis.uniswap_factory.getPool.call(
        env,
        verbs.utils.ZERO_ADDRESS,
        verbs.utils.hex_to_bytes(UNISWAP_V3_FACTORY),
        [WETH, DAI, fee],
    )[0][0]

    # Sanity check
    assert pool_address == UNISWAP_WETH_DAI.lower()
    pool_address = verbs.utils.hex_to_bytes(pool_address)

    # ------------------------
    # Initialize Uniswap agent
    # ------------------------
    agent = uniswap_agent_type(
        env=env,
        dt=0.01,
        fee=fee,
        i=10,  # idx of agent
        mu=mu,
        sigma=sigma,
        swap_router_abi=abis.swap_router,
        swap_router_address=swap_router_address,
        token_a_address=weth_address,
        token_b_address=dai_address,
        uniswap_pool_abi=abis.uniswap_pool,
        uniswap_pool_address=pool_address,
        quoter_abi=abis.quoter,
        quoter_address=quoter_address,
    )

    # mint and approve tokens for the Uniswap agent
    # - Mint DAI and WETH
    # - Approve the Swap Router to use these in their transactions
    mint_and_approve_weth(
        env=env,
        weth_abi=abis.weth_erc20,
        weth_address=weth_address,
        recipient=agent.address,
        contract_approved_address=swap_router_address,
        amount=int(1e24),
    )
    mint_and_approve_dai(
        env=env,
        dai_abi=abis.dai,
        dai_address=dai_address,
        contract_approved_address=swap_router_address,
        dai_admin_address=dai_admin_address,
        recipient=agent.address,
        amount=int(1e30),
    )

    # run simulation
    # - The Uniswap Agent records the price of the external market,
    #   and the price of Uniswap.
    agents = [agent]
    runner = verbs.sim.Sim(seed, env, agents)
    results = runner.run(n_steps=n_steps, show_progress=show_progress)

    return results


def init_cache(
    key: str,
    block_number: int,
    seed: int,
    n_steps: int,
    mu: float = 0.1,
    sigma: float = 0.6,
) -> verbs.types.Cache:
    """
    Generate a simulation request cache

    Run a simulation from a fork and store a cache of
    data request foe use in other simulations.

    Parameters
    ----------
    key: str
        Alchemy API key
    block_number: int
        Block number to fork from
    seed: int
        Random seed
    n_steps: int
        Number of simulation steps
    mu: float, optional
        GBM mu parameter, default 0.1
    sigma: float, optional
        GBM sigma parameter, default 0.6

    Returns
    -------
    cache: verbs.types.Cache
        Cache generated using :py:meth:`verbs.envs.ForkEnv.export_cache`.
    """

    # Fork environment from mainnet
    env = verbs.envs.ForkEnv(
        "https://eth-mainnet.g.alchemy.com/v2/{}".format(key),
        seed,
        block_number,
    )
    runner(
        env,
        seed,
        n_steps,
        mu=mu,
        sigma=sigma,
        uniswap_agent_type=partial(DummyUniswapAgent, sim_n_steps=n_steps),
    )

    cache = env.export_cache()

    with open(f"{PATH}/cache.json", "w") as f:
        json.dump(verbs.utils.cache_to_json(cache), f)

    return cache
