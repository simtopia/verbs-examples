import numpy as np
import verbs


class BorrowAgent:
    def __init__(
        self,
        network,
        i: int,
        pool_implementation_abi,
        oracle_abi,
        mintable_erc20_abi,
        pool_address: str,
        oracle_address: str,
        token_a_address: str,
        token_b_address: str,
        activation_rate: float,
    ):
        self.net = network
        self.address = verbs.utils.int_to_address(i)
        self.net.create_account(self.address, int(1e30))

        self.pool_implementation_abi = pool_implementation_abi
        self.pool_address = verbs.utils.hex_to_bytes(pool_address)
        self.oracle_address = verbs.utils.hex_to_bytes(oracle_address)
        self.oracle_abi = oracle_abi
        self.mintable_erc20_abi = mintable_erc20_abi

        self.token_a_address = verbs.utils.hex_to_bytes(
            token_a_address
        )  # collateral token - risky asset
        self.token_b_address = verbs.utils.hex_to_bytes(
            token_b_address
        )  # debt token - stablecoin

        self.decimals_token_b = mintable_erc20_abi.decimals.call(
            self.net, self.address, self.token_b_address, []
        )[0][0]

        self.has_borrowed = False
        self.has_supplied = False

        assert (
            0 < activation_rate and activation_rate < 1
        ), "activation_rate has to be between 0 and 1"
        self.activation_rate = activation_rate

    def update(self, rng: np.random.Generator, *args, **kwargs):
        balance_token_a = self.mintable_erc20_abi.balanceOf.call(
            self.net,
            self.address,
            self.token_a_address,
            [
                self.address,
            ],
        )[0][0]
        if rng.random() < self.activation_rate:
            if not self.has_supplied:
                supply_tx = self.pool_implementation_abi.supply.transaction(
                    self.address,
                    self.pool_address,
                    [self.token_a_address, 10**20, self.address, 0],
                )
                self.has_supplied = True
                return [supply_tx]
            elif not self.has_borrowed:
                user_data = self.pool_implementation_abi.getUserAccountData.call(
                    self.net, self.address, self.pool_address, [self.address]
                )[0]
                # available to borrow in base currency (in Aave, base currency is USD)
                available_borrow_base = user_data[2]

                # we convert the availble to borrow to borrow asset units
                borrow_asset_price = self.oracle_abi.getAssetPrice.call(
                    self.net, self.address, self.oracle_address, [self.token_b_address]
                )[0][0]
                coef = 10 ** (self.decimals_token_b - 4)
                u = rng.integers(low=9000, high=10000)
                available_borrow = int(
                    coef * available_borrow_base * u / borrow_asset_price
                )
                if available_borrow > 0:
                    borrow_tx = self.pool_implementation_abi.borrow.transaction(
                        self.address,
                        self.pool_address,
                        [self.token_b_address, available_borrow, 2, 0, self.address],
                    )
                    self.has_borrowed = True
                    return [borrow_tx]
                else:
                    return []
            else:
                return []
        else:
            return []

    def record(self):
        """Record the state of the agent"""
        user_data = self.pool_implementation_abi.getUserAccountData.call(
            self.net, self.address, self.pool_address, [self.address]
        )[0]
        health_factor = user_data[5]
        return health_factor