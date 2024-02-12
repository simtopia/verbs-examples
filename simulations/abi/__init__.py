from pathlib import Path

from verbs import abi

PATH = Path(__file__).parent

swap_router = abi.load_abi(f"{PATH}/../abi/SwapRouter.abi")
dai = abi.load_abi(f"{PATH}/../abi/dai.abi")
weth_erc20 = abi.load_abi(f"{PATH}/../abi/WETHMintableERC20.abi")
uniswap_pool = abi.load_abi(f"{PATH}/../abi/UniswapV3Pool.abi")
uniswap_factory = abi.load_abi(f"{PATH}/../abi/UniswapV3Factory.abi")

aave_pool = abi.load_abi(f"{PATH}/../abi/Pool-Implementation.abi")
aave_oracle = abi.load_abi(f"{PATH}/../abi/AaveOracle.abi")
quoter = abi.load_abi(f"{PATH}/../abi/Quoter_v2.abi")
uniswap_aggregator = abi.load_abi(f"{PATH}/../abi/UniswapAggregator.abi")
mock_aggregator = abi.load_abi(f"{PATH}/../abi/MockAggregator.abi")
aave_pool_addresses_provider = abi.load_abi(f"{PATH}/../abi/PoolAddressesProvider.abi")
aave_acl_manager = abi.load_abi(f"{PATH}/../abi/ACLManager.abi")
