"""
In this example we consider the interaction between Aave and Uniswap
via the following agents:

* A `Uniswap Agent` that trades between a Uniswap pool and
  an external market, modelled by a Geometric Brownian Motion,
  in order to make a profit.
* Several `Borrow Agents` that borrow from an Aave v3 pool.
* A `Liquidation Agent` that liquidated those positions from the
  `Borrow agents` that are in distress (that is, that their Health
  Factors are < 1) as long as the liquidation is profitable for
  the liquidation agent.

We consider the following:

* Uniswap v3 pool for WETH and DAI with fee 3000.
* `Borrow agents` borrow DAI and deposit WETH as collateral.
* The price of the risky asset (WETH) in terms of the stablecoin (DAI) in the
  external market is modelled by a GBM.
* The price of Uniswap follows the price in the external
  market. The Uniswap agent allows that by making the right trade in each step
  so that the new Uniswap price is the same as the price in the external market.
* The liquidator agent checks whether a liquidation is profitable before making
  the liquidation call.

Notes
-----
Profitability is checked by the following accountability:

* They check the amount of collateral that they would get by liquidating a
  fraction of a loan.
* They check the price of the trade in Uniswap necessary to close the short
  position in the debt asset.
* If they get a profit after closing their short position in the debt asset,
  then they make the transaction.

References
----------
#. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4540333
"""

import json
from functools import partial
from pathlib import Path
from typing import List

import verbs

from simulations import abi
from simulations.agents.borrow_agent import BorrowAgent
from simulations.agents.liquidation_agent import (
    AdversarialLiquidationAgent,
    LiquidationAgent,
)
from simulations.agents.uniswap_agent import DummyUniswapAgent, UniswapAgent
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

AAVE_DATA_PROVIDER = "0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3"
AAVE_POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
AAVE_ORACLE = "0x54586bE62E3c3580375aE3723C145253060Ca0C2"
AAVE_ADDRESS_PROVIDER = "0x2f39d218133AFaB8F2B819B1066c7E434Ad94E9e"
AAVE_ACL_MANAGER = "0xc2aaCf6553D20d1e9d78E365AAba8032af9c85b0"


def runner(
    env,
    seed: int,
    n_steps: int,
    *,
    n_borrow_agents: int,
    mu: float = 0.0,
    sigma: float = 0.3,
    adversarial_liquidator: bool = False,
    uniswap_agent_type=UniswapAgent,
    show_progress: bool = True,
) -> List[List]:
    """
    Aave simulation runner

    Parameters
    ----------
    env
        Simulation environment
    seed: int
        Random seed
    n_steps: int
        Number of simulation steps
    n_borrow_agents: int
        Number of simulated borrow agents
    mu: float, optional
        GBM mu parameter, default 0.0
    sigma: float, optional
        GBM sigma parameter, default 0.3
    adversarial_liquidator: bool, optional
        If ``True`` simulation will use an adversarial liquidator
        default ``False``
    show_progress: bool, optional
        If ``True`` simulation progress will be printed

    Returns
    -------
    list
        List of agent states recorded over the simulation
    """
    # Use uniswap_factory contract to get the address of WETH-DAI pool
    fee = 3000

    pool_address = abi.uniswap_factory.getPool.call(
        env,
        verbs.utils.ZERO_ADDRESS,
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
    quoter_address = verbs.utils.hex_to_bytes(UNISWAP_QUOTER)
    aave_pool_address = verbs.utils.hex_to_bytes(AAVE_POOL)
    uniswap_weth_dai = verbs.utils.hex_to_bytes(UNISWAP_WETH_DAI)
    aave_pool_address = verbs.utils.hex_to_bytes(AAVE_POOL)
    aave_address_provider = verbs.utils.hex_to_bytes(AAVE_ADDRESS_PROVIDER)
    aave_acl_manager_address = verbs.utils.hex_to_bytes(AAVE_ACL_MANAGER)
    aave_oracle_address = verbs.utils.hex_to_bytes(AAVE_ORACLE)

    # -------------------------
    # Initialize Uniswap agent
    # -------------------------
    uniswap_agent = uniswap_agent_type(
        env=env,
        dt=0.01,
        fee=fee,
        i=10,
        mu=mu,
        sigma=sigma,
        swap_router_abi=abi.swap_router,
        swap_router_address=swap_router_address,
        quoter_abi=abi.quoter,
        quoter_address=quoter_address,
        token_a_address=weth_address,
        token_b_address=dai_address,
        uniswap_pool_abi=abi.uniswap_pool,
        uniswap_pool_address=uniswap_weth_dai,
    )

    # Mint and approve tokens
    mint_and_approve_weth(
        env=env,
        weth_abi=abi.weth_erc20,
        weth_address=weth_address,
        recipient=uniswap_agent.address,
        contract_approved_address=swap_router_address,
        amount=int(1e24),
    )
    mint_and_approve_dai(
        env=env,
        dai_abi=abi.dai,
        dai_address=dai_address,
        contract_approved_address=swap_router_address,
        dai_admin_address=dai_admin_address,
        recipient=uniswap_agent.address,
        amount=int(1e30),
    )

    # -------------------------
    # Initialise borrow agents
    # -------------------------
    borrow_agents = [
        BorrowAgent(
            env=env,
            i=100 + i,
            pool_implementation_abi=abi.aave_pool,
            oracle_abi=abi.aave_oracle,
            mintable_erc20_abi=abi.weth_erc20,
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
        mint_and_approve_weth(
            env=env,
            weth_abi=abi.weth_erc20,
            weth_address=weth_address,
            recipient=borrow_agent.address,
            contract_approved_address=aave_pool_address,
            amount=int(1e24),
        )

    # ----------------------------------------------
    # Replace Chainlink with our price aggregation
    # ----------------------------------------------

    uniswap_aggregator_address = abi.uniswap_aggregator.constructor.deploy(
        env,
        verbs.utils.ZERO_ADDRESS,
        abi.UNISWAP_AGGREGATOR_BYTECODE,
        [
            uniswap_weth_dai,
            weth_address,
            dai_address,
        ],
    )

    # We load the dummy Mock Aggregator contract that keeps the price of a
    # token constant (that will be our numeraire)
    mock_aggregator_address = abi.mock_aggregator.constructor.deploy(
        env, verbs.utils.ZERO_ADDRESS, abi.MOCK_AGGREGATOR_BYTECODE, [10**8]
    )

    aave_acl_admin = abi.aave_pool_addresses_provider.getACLAdmin.call(
        env, verbs.utils.ZERO_ADDRESS, aave_address_provider, []
    )[0][0]
    aave_acl_admin_address = verbs.utils.hex_to_bytes(aave_acl_admin)

    pool_admin_role = abi.aave_acl_manager.POOL_ADMIN_ROLE.call(
        env,
        aave_acl_admin_address,
        aave_acl_manager_address,
        [],
    )[0][0]

    abi.aave_acl_manager.grantRole.execute(
        env,
        aave_acl_admin_address,
        aave_acl_manager_address,
        [
            pool_admin_role,
            aave_acl_admin_address,
        ],
    )

    abi.aave_oracle.setAssetSources.execute(
        env,
        aave_acl_admin_address,
        verbs.utils.hex_to_bytes(AAVE_ORACLE),
        [
            [weth_address, dai_address],
            [uniswap_aggregator_address, mock_aggregator_address],
        ],
    )

    # -----------------------------
    # Initialise liquidation agent
    # -----------------------------
    liquidation_agent_type = (
        partial(
            AdversarialLiquidationAgent,
            aave_oracle_abi=abi.aave_oracle,
            aave_oracle_address=aave_oracle_address,
        )
        if adversarial_liquidator
        else LiquidationAgent
    )
    liquidation_agent = liquidation_agent_type(
        env=env,
        i=1000,
        pool_implementation_abi=abi.aave_pool,
        mintable_erc20_abi=abi.weth_erc20,
        pool_address=aave_pool_address,
        token_a_address=weth_address,
        token_b_address=dai_address,
        liquidation_addresses=[borrow_agent.address for borrow_agent in borrow_agents],
        uniswap_pool_abi=abi.uniswap_pool,
        quoter_abi=abi.quoter,
        swap_router_abi=abi.swap_router,
        uniswap_pool_address=uniswap_weth_dai,
        quoter_address=verbs.utils.hex_to_bytes(UNISWAP_QUOTER),
        swap_router_address=swap_router_address,
        uniswap_fee=fee,
    )

    mint_and_approve_weth(
        env=env,
        weth_abi=abi.weth_erc20,
        weth_address=weth_address,
        recipient=liquidation_agent.address,
        contract_approved_address=swap_router_address,
        amount=int(1e30),
    )
    mint_and_approve_dai(
        env=env,
        dai_abi=abi.dai,
        dai_address=dai_address,
        contract_approved_address=aave_pool_address,
        dai_admin_address=dai_admin_address,
        recipient=liquidation_agent.address,
        amount=int(1e35),
    )

    # Run simulation
    agents = [uniswap_agent] + borrow_agents + [liquidation_agent]

    runner = verbs.sim.Sim(seed, env, agents)
    results = runner.run(n_steps=n_steps, show_progress=show_progress)

    return results


def init_cache(
    key: str,
    block_number: int,
    seed: int,
    n_steps: int,
    n_borrow_agents: int,
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
    n_borrow_agents: int
        Number of simulated borrow agents
    mu: float, optional
        GBM mu parameter, default 0.1
    sigma: float, optional
        GBM sigma parameter, default 0.6

    Returns
    -------
    verbs.types.Cache
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
        n_borrow_agents=n_borrow_agents,
        sigma=sigma,
        mu=mu,
        uniswap_agent_type=partial(DummyUniswapAgent, sim_n_steps=n_steps),
    )
    cache = env.export_cache()

    with open(f"{PATH}/cache.json", "w") as f:
        json.dump(verbs.utils.cache_to_json(cache), f)

    return cache
