"""
In this example
"""


import argparse

import numpy as np
import verbs

from agents.uniswap_agent import UniswapAgent

WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
DAI_ADMIN = "0x9759A6Ac90977b93B58547b4A71c78317f391A28"
UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
UNISWAP_WETH_DAI = "0xC2e9F25Be6257c210d7Adf0D4Cd6E3E881ba25f8"  # sanity check, obtained from the factory contract using web3.py
SWAP_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, help="prive key from Alchemy")
    parser.add_argument(
        "--block", type=int, default=18784000, help="Ethereum block number"
    )

    args = parser.parse_args()
    key = args.key
    block_number = args.block

    swap_router_abi = verbs.abi.load_abi("abi/SwapRouter.abi")
    dai_abi = verbs.abi.load_abi("abi/dai.abi")
    weth_erc20_abi = verbs.abi.load_abi("abi/WETHMintableERC20.abi")
    uniswap_pool_abi = verbs.abi.load_abi("abi/UniswapV3Pool.abi")
    uniswap_factory_abi = verbs.abi.load_abi("abi/UniswapV3Factory.abi")

    net = verbs.envs.ForkEnv(
        "https://eth-mainnet.g.alchemy.com/v2/{}".format(key),
        0,
        block_number,
        "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
    )

    fee = 3000
    get_pool_args = uniswap_factory_abi.getPool.encode([WETH, DAI, fee])
    pool_address = uniswap_factory_abi.getPool.call(
        net,
        net.admin_address,
        verbs.utils.hex_to_bytes(UNISWAP_V3_FACTORY),
        [WETH, DAI, fee],
    )[0][0]

    # Sanity check
    assert pool_address == UNISWAP_WETH_DAI.lower()

    # Initialize Uniswap agent
    agent = UniswapAgent(
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
        sender=agent.address,
        value=int(1e24),
    )

    weth_erc20_abi.approve.execute(
        sender=agent.address,
        address=verbs.utils.hex_to_bytes(WETH),
        env=net,
        args=[verbs.utils.hex_to_bytes(SWAP_ROUTER), int(1e24)],
    )

    dai_abi.mint.execute(
        address=verbs.utils.hex_to_bytes(DAI),
        sender=verbs.utils.hex_to_bytes(DAI_ADMIN),
        env=net,
        args=[agent.address, int(1e30)],
    )

    dai_abi.approve.execute(
        sender=agent.address,
        address=verbs.utils.hex_to_bytes(DAI),
        env=net,
        args=[verbs.utils.hex_to_bytes(SWAP_ROUTER), int(1e30)],
    )

    # run simulation
    runner = verbs.sim.Sim(101, net, [agent])
    results = runner.run(n_steps=100)
