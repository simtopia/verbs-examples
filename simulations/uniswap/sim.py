"""
In this example we model an agent that trades between a Uniswap pool and
and an external market, modelled by a Geometric Brownian Motion, in order
to make a profit.

    - We consider the Uniswap v3 pool for WETH and DAI with fee 3000.
    - The price of the risky asset (WETH) in terms of the stablecoin (DAI) in the
      external market is modelled by a GBM.
    - The goal of the simulation is for the price of Uniswap to follow the price
      in the external market. The Uniswap agent takes of that in each step, by
      making the right trade so that the new Uniswap price is the same as the
      price in the external market.
"""

import json
from pathlib import Path

import verbs

from simulations.agents.uniswap_agent import UniswapAgent

PATH = Path(__file__).parent

WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
DAI_ADMIN = "0x9759A6Ac90977b93B58547b4A71c78317f391A28"
UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
# sanity check, obtained from the factory contract using web3.py
UNISWAP_WETH_DAI = "0xC2e9F25Be6257c210d7Adf0D4Cd6E3E881ba25f8"
SWAP_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"


def runner(seed: int, n_steps: int, env):

    # ABIs
    swap_router_abi = verbs.abi.load_abi(f"{PATH}/../abi/SwapRouter.abi")
    dai_abi = verbs.abi.load_abi(f"{PATH}/../abi/dai.abi")
    weth_erc20_abi = verbs.abi.load_abi(f"{PATH}/../abi/WETHMintableERC20.abi")
    uniswap_pool_abi = verbs.abi.load_abi(f"{PATH}/../abi/UniswapV3Pool.abi")
    uniswap_factory_abi = verbs.abi.load_abi(f"{PATH}/../abi/UniswapV3Factory.abi")

    # Convert addresses
    weth_address = verbs.utils.hex_to_bytes(WETH)
    dai_address = verbs.utils.hex_to_bytes(DAI)
    swap_router_address = verbs.utils.hex_to_bytes(SWAP_ROUTER)

    # Example: Use uniswap_factory contract to get the address of WETH-DAI
    # pool with fee 3000
    fee = 3000

    pool_address = uniswap_factory_abi.getPool.call(
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
    agent = UniswapAgent(
        env=env,
        dt=0.01,
        fee=fee,
        i=10,  # idx of agent
        mu=0.0,
        sigma=0.3,
        swap_router_abi=swap_router_abi,
        swap_router_address=swap_router_address,
        token_a_address=weth_address,
        token_b_address=dai_address,
        uniswap_pool_abi=uniswap_pool_abi,
        uniswap_pool_address=pool_address,
    )

    # mint and approve tokens for the Uniswap agent
    # - Mint DAI and WETH
    # - Approve the Swap Router to use these in their transactions
    weth_erc20_abi.deposit.execute(
        address=weth_address,
        args=[],
        env=env,
        sender=agent.address,
        value=int(1e24),
    )

    weth_erc20_abi.approve.execute(
        sender=agent.address,
        address=weth_address,
        env=env,
        args=[swap_router_address, int(1e24)],
    )

    dai_abi.mint.execute(
        address=dai_address,
        sender=verbs.utils.hex_to_bytes(DAI_ADMIN),
        env=env,
        args=[agent.address, int(1e30)],
    )

    dai_abi.approve.execute(
        sender=agent.address,
        address=dai_address,
        env=env,
        args=[swap_router_address, int(1e30)],
    )

    # run simulation
    # - The Uniswap Agent records the price of the external market,
    #   and the price of Uniswap.
    runner = verbs.sim.Sim(seed, env, [agent])
    results = runner.run(n_steps=n_steps)

    return env, results


def init_cache(key: str, block_number: int, seed: int, n_steps: int):

    # Fork environment from mainnet
    env = verbs.envs.ForkEnv(
        "https://eth-mainnet.g.alchemy.com/v2/{}".format(key),
        seed,
        block_number,
    )

    env, _ = runner(seed, n_steps, env)

    return env.export_cache()


def run_from_cache(seed: int, n_steps: int):

    with open(f"{PATH}/cache.json", "r") as f:
        cache_json = json.load(f)

    cache = verbs.utils.cache_from_json(cache_json)

    env = verbs.envs.EmptyEnv(seed, cache=cache)

    _, results = runner(seed, n_steps, env)

    return results
