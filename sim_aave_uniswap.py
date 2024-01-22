"""
In this example
"""
import argparse
import json

import verbs

from agents.admin_agent import AdminAgent
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, help="prive key from Alchemy")
    parser.add_argument(
        "--block", type=int, default=18784000, help="Ethereum block number"
    )
    parser.add_argument(
        "--n_borrow_agents", type=int, default=10, help="Number of borrowing agents"
    )

    args = parser.parse_args()
    key = args.key
    block_number = args.block
    n_borrow_agents = args.n_borrow_agents

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

    # Fork mainnet
    net = verbs.envs.ForkEnv(
        "https://eth-mainnet.g.alchemy.com/v2/{}".format(key),
        0,
        block_number,
    )

    # admin agent
    admin_agent = AdminAgent(net, i=1)

    # # Use uniswap_factory contract to get the address of WETH-DAI pool
    fee = 3000
    get_pool_args = uniswap_factory_abi.getPool.encode([WETH, DAI, fee])
    pool_address = uniswap_factory_abi.getPool.call(
        net,
        admin_agent.address,
        verbs.utils.hex_to_bytes(UNISWAP_V3_FACTORY),
        [WETH, DAI, fee],
    )[0][0]

    # Sanity check
    # assert pool_address == UNISWAP_WETH_DAI.lower()

    ###########################
    # Initialize Uniswap agent
    ###########################
    uniswap_agent = UniswapAgent(
        network=net,
        dt=0.01,
        fee=fee,
        i=10,
        mu=0.0,
        sigma=0.3,
        swap_router_abi=swap_router_abi,
        swap_router_address=SWAP_ROUTER,
        token_a_address=WETH,
        token_a_price=2000.0,
        token_b_address=DAI,
        token_b_price=1.0,
        uniswap_pool_abi=uniswap_pool_abi,
        uniswap_pool_address=pool_address,
    )

    # mint and approve tokens
    weth_erc20_abi.deposit.execute(
        address=verbs.utils.hex_to_bytes(WETH),
        args=[],
        env=net,
        sender=uniswap_agent.address,
        value=int(1e24),
    )

    weth_erc20_abi.approve.execute(
        sender=uniswap_agent.address,
        address=verbs.utils.hex_to_bytes(WETH),
        env=net,
        args=[verbs.utils.hex_to_bytes(SWAP_ROUTER), int(1e24)],
    )

    dai_abi.mint.execute(
        address=verbs.utils.hex_to_bytes(DAI),
        sender=verbs.utils.hex_to_bytes(DAI_ADMIN),
        env=net,
        args=[uniswap_agent.address, int(1e30)],
    )

    dai_abi.approve.execute(
        sender=uniswap_agent.address,
        address=verbs.utils.hex_to_bytes(DAI),
        env=net,
        args=[verbs.utils.hex_to_bytes(SWAP_ROUTER), int(1e30)],
    )

    ##############################
    # Initialise borrow agents
    ##############################
    borrow_agents = [
        BorrowAgent(
            network=net,
            i=100 + i,
            pool_implementation_abi=aave_pool_abi,
            oracle_abi=aave_oracle_abi,
            mintable_erc20_abi=weth_erc20_abi,
            pool_address=AAVE_POOL,
            oracle_address=AAVE_ORACLE,
            token_a_address=WETH,
            token_b_address=DAI,
            activation_rate=0.5,
        )
        for i in range(n_borrow_agents)
    ]

    # Mint WETH and approve WETH
    for borrow_agent in borrow_agents:
        weth_erc20_abi.deposit.execute(
            address=verbs.utils.hex_to_bytes(WETH),
            args=[],
            env=net,
            sender=borrow_agent.address,
            value=int(1e24),
        )

        weth_erc20_abi.approve.execute(
            sender=borrow_agent.address,
            address=verbs.utils.hex_to_bytes(WETH),
            env=net,
            args=[verbs.utils.hex_to_bytes(AAVE_POOL), int(1e24)],
        )

    ################################
    # Initialise liquidation agent
    ################################
    liquidation_agent = LiquidationAgent(
        network=net,
        i=1000,
        pool_implementation_abi=aave_pool_abi,
        mintable_erc20_abi=weth_erc20_abi,
        pool_address=AAVE_POOL,
        token_a_address=WETH,
        token_b_address=DAI,
        liquidation_addresses=[borrow_agent.address for borrow_agent in borrow_agents],
        uniswap_pool_abi=uniswap_pool_abi,
        quoter_abi=quoter_abi,
        swap_router_abi=swap_router_abi,
        uniswap_pool_address=UNISWAP_WETH_DAI,
        quoter_address=UNISWAP_QUOTER,
        swap_router_address=SWAP_ROUTER,
        uniswap_fee=fee,
    )
    weth_erc20_abi.deposit.execute(
        address=verbs.utils.hex_to_bytes(WETH),
        args=[],
        env=net,
        sender=liquidation_agent.address,
        value=int(1e24),
    )

    weth_erc20_abi.approve.execute(
        sender=liquidation_agent.address,
        address=verbs.utils.hex_to_bytes(WETH),
        env=net,
        args=[verbs.utils.hex_to_bytes(SWAP_ROUTER), int(1e24)],
    )

    dai_abi.mint.execute(
        address=verbs.utils.hex_to_bytes(DAI),
        sender=verbs.utils.hex_to_bytes(DAI_ADMIN),
        env=net,
        args=[liquidation_agent.address, int(1e30)],
    )

    dai_abi.approve.execute(
        sender=liquidation_agent.address,
        address=verbs.utils.hex_to_bytes(DAI),
        env=net,
        args=[verbs.utils.hex_to_bytes(AAVE_POOL), int(1e30)],
    )

    ################################################
    # Replace Chainlink with our price aggregation
    ################################################

    # We load the Uniswap Aggregator contract that gets the price from the Uniswap pool
    with open("abi/UniswapAggregator.json", "r") as f:
        uniswap_aggregator_contract = json.load(f)

    uniswap_aggregator_address = uniswap_aggregator_abi.constructor.deploy(
        net,
        admin_agent.address,
        uniswap_aggregator_contract["bytecode"],
        [
            verbs.utils.hex_to_bytes(UNISWAP_WETH_DAI),
            verbs.utils.hex_to_bytes(WETH),
            verbs.utils.hex_to_bytes(DAI),
        ],
    )

    # We load the dummy Mock Aggregator contract that keeps the price of a
    # token constant (that will be our numeraire)
    with open("abi/MockAggregator.json", "r") as f:
        mock_aggregator_contract = json.load(f)
    mock_aggregator_address = mock_aggregator_abi.constructor.deploy(
        net, admin_agent.address, mock_aggregator_contract["bytecode"], [10**8]
    )

    aave_acl_admin = aave_pool_addresses_provider_abi.getACLAdmin.call(
        net, admin_agent.address, verbs.utils.hex_to_bytes(AAVE_ADDRESS_PROVIDER), []
    )[0][0]

    pool_admin_role = aave_acl_manager_abi.POOL_ADMIN_ROLE.call(
        net,
        verbs.utils.hex_to_bytes(aave_acl_admin),
        verbs.utils.hex_to_bytes(AAVE_ACL_MANAGER),
        [],
    )[0][0]

    aave_acl_manager_abi.grantRole.execute(
        net,
        verbs.utils.hex_to_bytes(aave_acl_admin),
        verbs.utils.hex_to_bytes(AAVE_ACL_MANAGER),
        [
            pool_admin_role,
            verbs.utils.hex_to_bytes(aave_acl_admin),
        ],
    )

    aave_oracle_abi.setAssetSources.execute(
        net,
        verbs.utils.hex_to_bytes(aave_acl_admin),
        verbs.utils.hex_to_bytes(AAVE_ORACLE),
        [
            [verbs.utils.hex_to_bytes(WETH), verbs.utils.hex_to_bytes(DAI)],
            [uniswap_aggregator_address, mock_aggregator_address],
        ],
    )

    # run simulation
    agents = [uniswap_agent] + borrow_agents + [liquidation_agent]
    runner = verbs.sim.Sim(101, net, agents)
    results = runner.run(n_steps=100)
