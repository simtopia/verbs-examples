"""
In this example we consider the interaction between Aave and Uniswap
via the following agents:

    1. A `Uniswap Agent` that trades between a Uniswap pool and
       an external market, modelled by a Geometric Brownian Motion,
       in order to make a profit.

    2. Several `Borrow Agents` that borrow from an Aave v3 pool.

    3. A `Liquidation Agent` that liquidated those positions from the
       `Borrow agents` that are in distress (that is, that their Health
       Factors are < 1) as long as the liquidation is profitable for
       the liquidation agent.

We consider the following pools and tokens:

    - Uniswap v3 pool for WETH and DAI with fee 3000.

    - `Borrow agents` borrow DAI and deposit WETH as collateral.

    - The price of the risky asset (WETH) in terms of the stablecoin (DAI) in the
      external market is modelled by a GBM.

    - The price of Uniswap follows the price in the external
      market. The Uniswap agent allows that by making the right trade in each step
      so that the new Uniswap price is the same as the price in the external market.

    - The liquidator agent checks whether a liquidation is profitable before making
      the liquidation call:
        - They check the amount of collateral that they would get by liquidation a
          fraction of a loan.
        - They check the price of the trade in Uniswap necessary to close the short
          position in the debt asset.
        - If they get a profit after closing their short position in the debt asset,
          then they make the transaction.


Reference: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4540333
"""


import argparse
import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import verbs

from agents import ZERO_ADDRESS
from agents.borrow_agent import BorrowAgent
from agents.liquidation_agent import LiquidationAgent
from agents.uniswap_agent import UniswapAgent

WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
DAI_ADMIN = "0x9759A6Ac90977b93B58547b4A71c78317f391A28"
UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
# sanity check, obtained from the factory contract using web3.py
UNISWAP_WETH_DAI = "0xC2e9F25Be6257c210d7Adf0D4Cd6E3E881ba25f8"
SWAP_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
UNISWAP_QUOTER = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"

AAVE_DATA_PROVIDER = "0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3"
AAVE_POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
AAVE_ORACLE = "0x54586bE62E3c3580375aE3723C145253060Ca0C2"
AAVE_ADDRESS_PROVIDER = "0x2f39d218133AFaB8F2B819B1066c7E434Ad94E9e"
AAVE_ACL_MANAGER = "0xc2aaCf6553D20d1e9d78E365AAba8032af9c85b0"


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="AAVE agent-based simulation")
    parser.add_argument("key", type=str, help="Alchemy API key")
    parser.add_argument(
        "--block", type=int, default=18784000, help="Ethereum block number"
    )
    parser.add_argument(
        "--n_borrow_agents", type=int, default=2, help="Number of borrowing agents"
    )
    parser.add_argument("--sigma", type=float, default=0.3, help="price volatility")
    parser.add_argument(
        "--n_steps", type=int, default=100, help="Number of steps of the simulation"
    )

    args = parser.parse_args()
    key = args.key
    block_number = args.block
    n_borrow_agents = args.n_borrow_agents
    sigma = args.sigma
    n_steps = args.n_steps

    # ABIs
    swap_router_abi = verbs.abi.load_abi("abi/SwapRouter.abi")
    dai_abi = verbs.abi.load_abi("abi/dai.abi")
    weth_erc20_abi = verbs.abi.load_abi("abi/WETHMintableERC20.abi")
    uniswap_pool_abi = verbs.abi.load_abi("abi/UniswapV3Pool.abi")
    uniswap_factory_abi = verbs.abi.load_abi("abi/UniswapV3Factory.abi")
    aave_pool_abi = verbs.abi.load_abi("abi/Pool-Implementation.abi")
    aave_oracle_abi = verbs.abi.load_abi("abi/AaveOracle.abi")
    quoter_abi = verbs.abi.load_abi("abi/Quoter_v2.abi")
    uniswap_aggregator_abi = verbs.abi.load_abi("abi/UniswapAggregator.abi")
    mock_aggregator_abi = verbs.abi.load_abi("abi/MockAggregator.abi")
    aave_pool_addresses_provider_abi = verbs.abi.load_abi(
        "abi/PoolAddressesProvider.abi"
    )
    aave_acl_manager_abi = verbs.abi.load_abi("abi/ACLManager.abi")

    # Fork environment from mainnet
    env = verbs.envs.ForkEnv(
        "https://eth-mainnet.g.alchemy.com/v2/{}".format(key),
        0,
        block_number,
    )

    # Use uniswap_factory contract to get the address of WETH-DAI pool
    fee = 3000
    get_pool_args = uniswap_factory_abi.getPool.encode([WETH, DAI, fee])
    pool_address = uniswap_factory_abi.getPool.call(
        env,
        ZERO_ADDRESS,
        verbs.utils.hex_to_bytes(UNISWAP_V3_FACTORY),
        [WETH, DAI, fee],
    )[0][0]

    # Sanity check
    assert pool_address == UNISWAP_WETH_DAI.lower()

    # Convert addresses to bytes
    weth_address = verbs.utils.hex_to_bytes(WETH)
    dai_address = verbs.utils.hex_to_bytes(DAI)
    dai_admin_address = verbs.utils.hex_to_bytes(DAI_ADMIN)
    swap_router_address = verbs.utils.hex_to_bytes(SWAP_ROUTER)
    aave_pool_address = verbs.utils.hex_to_bytes(AAVE_POOL)
    uniswap_weth_dai = verbs.utils.hex_to_bytes(UNISWAP_WETH_DAI)
    aave_pool_address = verbs.utils.hex_to_bytes(AAVE_POOL)
    aave_address_provider = verbs.utils.hex_to_bytes(AAVE_ADDRESS_PROVIDER)
    aave_acl_manager_address = verbs.utils.hex_to_bytes(AAVE_ACL_MANAGER)
    aave_oracle_address = verbs.utils.hex_to_bytes(AAVE_ORACLE)

    # -------------------------
    # Initialize Uniswap agent
    # -------------------------
    uniswap_agent = UniswapAgent(
        env=env,
        dt=0.01,
        fee=fee,
        i=10,
        mu=0.0,
        sigma=sigma,
        swap_router_abi=swap_router_abi,
        swap_router_address=SWAP_ROUTER,
        token_a_address=WETH,
        token_b_address=DAI,
        uniswap_pool_abi=uniswap_pool_abi,
        uniswap_pool_address=pool_address,
    )

    # Mint and approve tokens
    weth_erc20_abi.deposit.execute(
        address=weth_address,
        args=[],
        env=env,
        sender=uniswap_agent.address,
        value=int(1e24),
    )

    weth_erc20_abi.approve.execute(
        sender=uniswap_agent.address,
        address=weth_address,
        env=env,
        args=[swap_router_address, int(1e24)],
    )

    dai_abi.mint.execute(
        address=dai_address,
        sender=dai_admin_address,
        env=env,
        args=[uniswap_agent.address, int(1e30)],
    )

    dai_abi.approve.execute(
        sender=uniswap_agent.address,
        address=dai_address,
        env=env,
        args=[swap_router_address, int(1e30)],
    )

    ##############################
    # Initialise borrow agents
    ##############################
    borrow_agents = [
        BorrowAgent(
            env=env,
            i=100 + i,
            pool_implementation_abi=aave_pool_abi,
            oracle_abi=aave_oracle_abi,
            mintable_erc20_abi=weth_erc20_abi,
            pool_address=aave_pool_address,
            oracle_address=aave_oracle_address,
            token_a_address=weth_address,
            token_b_address=dai_address,
            activation_rate=0.5,
        )
        for i in range(n_borrow_agents)
    ]

    # Mint WETH and approve WETH
    for borrow_agent in borrow_agents:
        weth_erc20_abi.deposit.execute(
            address=weth_address,
            args=[],
            env=env,
            sender=borrow_agent.address,
            value=int(1e24),
        )

        weth_erc20_abi.approve.execute(
            sender=borrow_agent.address,
            address=weth_address,
            env=env,
            args=[aave_pool_address, int(1e24)],
        )

    ################################
    # Initialise liquidation agent
    ################################
    liquidation_agent = LiquidationAgent(
        env=env,
        i=1000,
        pool_implementation_abi=aave_pool_abi,
        mintable_erc20_abi=weth_erc20_abi,
        pool_address=aave_pool_address,
        token_a_address=weth_address,
        token_b_address=dai_address,
        liquidation_addresses=[borrow_agent.address for borrow_agent in borrow_agents],
        uniswap_pool_abi=uniswap_pool_abi,
        quoter_abi=quoter_abi,
        swap_router_abi=swap_router_abi,
        uniswap_pool_address=uniswap_weth_dai,
        quoter_address=verbs.utils.hex_to_bytes(UNISWAP_QUOTER),
        swap_router_address=swap_router_address,
        uniswap_fee=fee,
    )

    weth_erc20_abi.deposit.execute(
        address=weth_address,
        args=[],
        env=env,
        sender=liquidation_agent.address,
        value=int(1e30),
    )

    weth_erc20_abi.approve.execute(
        sender=liquidation_agent.address,
        address=weth_address,
        env=env,
        args=[swap_router_address, int(1e30)],
    )

    dai_abi.mint.execute(
        address=dai_address,
        sender=dai_admin_address,
        env=env,
        args=[liquidation_agent.address, int(1e35)],
    )

    dai_abi.approve.execute(
        sender=liquidation_agent.address,
        address=dai_address,
        env=env,
        args=[aave_pool_address, int(1e35)],
    )

    # ----------------------------------------------
    # Replace Chainlink with our price aggregation
    # ----------------------------------------------

    # We load the Uniswap Aggregator contract that gets the price from the Uniswap pool
    with open("abi/UniswapAggregator.json", "r") as f:
        uniswap_aggregator_contract = json.load(f)

    uniswap_aggregator_address = uniswap_aggregator_abi.constructor.deploy(
        env,
        ZERO_ADDRESS,
        uniswap_aggregator_contract["bytecode"],
        [
            uniswap_weth_dai,
            weth_address,
            dai_address,
        ],
    )

    # We load the dummy Mock Aggregator contract that keeps the price of a
    # token constant (that will be our numeraire)
    with open("abi/MockAggregator.json", "r") as f:
        mock_aggregator_contract = json.load(f)

    mock_aggregator_address = mock_aggregator_abi.constructor.deploy(
        env, ZERO_ADDRESS, mock_aggregator_contract["bytecode"], [10**8]
    )

    aave_acl_admin = aave_pool_addresses_provider_abi.getACLAdmin.call(
        env, ZERO_ADDRESS, aave_address_provider, []
    )[0][0]
    aave_acl_admin_address = verbs.utils.hex_to_bytes(aave_acl_admin)

    pool_admin_role = aave_acl_manager_abi.POOL_ADMIN_ROLE.call(
        env,
        aave_acl_admin_address,
        aave_acl_manager_address,
        [],
    )[0][0]

    aave_acl_manager_abi.grantRole.execute(
        env,
        aave_acl_admin_address,
        aave_acl_manager_address,
        [
            pool_admin_role,
            aave_acl_admin_address,
        ],
    )

    aave_oracle_abi.setAssetSources.execute(
        env,
        aave_acl_admin_address,
        verbs.utils.hex_to_bytes(AAVE_ORACLE),
        [
            [weth_address, dai_address],
            [uniswap_aggregator_address, mock_aggregator_address],
        ],
    )

    # Run simulation
    agents = [uniswap_agent] + borrow_agents + [liquidation_agent]
    runner = verbs.sim.Sim(10, env, agents)
    results = runner.run(n_steps=n_steps)

    plot_results(results, n_borrow_agents)
