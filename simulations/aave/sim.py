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

import json
from pathlib import Path

import verbs

from simulations.agents.borrow_agent import BorrowAgent
from simulations.agents.liquidation_agent import LiquidationAgent
from simulations.agents.uniswap_agent import UniswapAgent

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


def runner(seed: int, n_steps: int, n_borrow_agents: int, env, sigma: float):

    # ABIs
    swap_router_abi = verbs.abi.load_abi(f"{PATH}/../abi/SwapRouter.abi")
    dai_abi = verbs.abi.load_abi(f"{PATH}/../abi/dai.abi")
    weth_erc20_abi = verbs.abi.load_abi(f"{PATH}/../abi/WETHMintableERC20.abi")
    uniswap_pool_abi = verbs.abi.load_abi(f"{PATH}/../abi/UniswapV3Pool.abi")
    uniswap_factory_abi = verbs.abi.load_abi(f"{PATH}/../abi/UniswapV3Factory.abi")
    aave_pool_abi = verbs.abi.load_abi(f"{PATH}/../abi/Pool-Implementation.abi")
    aave_oracle_abi = verbs.abi.load_abi(f"{PATH}/../abi/AaveOracle.abi")
    quoter_abi = verbs.abi.load_abi(f"{PATH}/../abi/Quoter_v2.abi")
    uniswap_aggregator_abi = verbs.abi.load_abi(f"{PATH}/../abi/UniswapAggregator.abi")
    mock_aggregator_abi = verbs.abi.load_abi(f"{PATH}/../abi/MockAggregator.abi")
    aave_pool_addresses_provider_abi = verbs.abi.load_abi(
        f"{PATH}/../abi/PoolAddressesProvider.abi"
    )
    aave_acl_manager_abi = verbs.abi.load_abi(f"{PATH}/../abi/ACLManager.abi")

    # Use uniswap_factory contract to get the address of WETH-DAI pool
    fee = 3000

    pool_address = uniswap_factory_abi.getPool.call(
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
        swap_router_address=swap_router_address,
        token_a_address=weth_address,
        token_b_address=dai_address,
        uniswap_pool_abi=uniswap_pool_abi,
        uniswap_pool_address=uniswap_weth_dai,
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
    with open(f"{PATH}/../abi/UniswapAggregator.json", "r") as f:
        uniswap_aggregator_contract = json.load(f)

    uniswap_aggregator_address = uniswap_aggregator_abi.constructor.deploy(
        env,
        verbs.utils.ZERO_ADDRESS,
        uniswap_aggregator_contract["bytecode"],
        [
            uniswap_weth_dai,
            weth_address,
            dai_address,
        ],
    )

    # We load the dummy Mock Aggregator contract that keeps the price of a
    # token constant (that will be our numeraire)
    with open(f"{PATH}/../abi/MockAggregator.json", "r") as f:
        mock_aggregator_contract = json.load(f)

    mock_aggregator_address = mock_aggregator_abi.constructor.deploy(
        env, verbs.utils.ZERO_ADDRESS, mock_aggregator_contract["bytecode"], [10**8]
    )

    aave_acl_admin = aave_pool_addresses_provider_abi.getACLAdmin.call(
        env, verbs.utils.ZERO_ADDRESS, aave_address_provider, []
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
    runner = verbs.sim.Sim(seed, env, agents)
    results = runner.run(n_steps=n_steps)

    return env, results


def init_cache(
    key: str,
    block_number: int,
    seed: int,
    n_steps: int,
    n_borrow_agents: int,
    sigma: float,
):

    # Fork environment from mainnet
    env = verbs.envs.ForkEnv(
        "https://eth-mainnet.g.alchemy.com/v2/{}".format(key),
        seed,
        block_number,
    )

    env, _ = runner(seed, n_steps, n_borrow_agents, env, sigma)

    return env.export_cache()


def run_from_cache(seed: int, n_steps: int, n_borrow_agents: int, sigma: float):

    with open(f"{PATH}/cache.json", "r") as f:
        cache_json = json.load(f)

    cache = verbs.utils.cache_from_json(cache_json)

    env = verbs.envs.EmptyEnv(seed, cache=cache)

    _, results = runner(seed, n_steps, n_borrow_agents, env, sigma)

    return results
