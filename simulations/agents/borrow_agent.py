import numpy as np
import verbs


class BorrowAgent:
    def __init__(
        self,
        env,
        i: int,
        pool_implementation_abi,
        oracle_abi,
        mintable_erc20_abi,
        pool_address: bytes,
        oracle_address: bytes,
        token_a_address: bytes,
        token_b_address: bytes,
        activation_rate: float,
    ):
        self.address = verbs.utils.int_to_address(i)
        env.create_account(self.address, int(1e30))

        self.pool_implementation_abi = pool_implementation_abi
        self.pool_address = pool_address
        self.oracle_address = oracle_address
        self.oracle_abi = oracle_abi
        self.mintable_erc20_abi = mintable_erc20_abi

        # collateral token - risky asset
        self.token_a_address = token_a_address
        # debt token - stablecoin
        self.token_b_address = token_b_address

        self.decimals_token_b = mintable_erc20_abi.decimals.call(
            env, self.address, self.token_b_address, []
        )[0][0]

        self.has_borrowed = False
        self.has_supplied = False

        assert (
            0 < activation_rate and activation_rate < 1
        ), "activation_rate has to be between 0 and 1"
        self.activation_rate = activation_rate

        self.step = 0

    def update(self, rng: np.random.Generator, env):
        self.step += 1

        if rng.random() < self.activation_rate:
            if not self.has_supplied:
                supply_tx = self.pool_implementation_abi.supply.transaction(
                    self.address,
                    self.pool_address,
                    [self.token_a_address, 10**25, self.address, 0],
                )
                self.has_supplied = True
                return [supply_tx]
            elif not self.has_borrowed:
                user_data = self.pool_implementation_abi.getUserAccountData.call(
                    env, self.address, self.pool_address, [self.address]
                )[0]
                # available to borrow in base currency (in Aave, base currency is USD)
                available_borrow_base = user_data[2]

                # we convert the available to borrow to borrow asset units
                borrow_asset_price = self.oracle_abi.getAssetPrice.call(
                    env, self.address, self.oracle_address, [self.token_b_address]
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

    def record(self, env):
        """Record the state of the agent"""
        user_data = self.pool_implementation_abi.getUserAccountData.call(
            env, self.address, self.pool_address, [self.address]
        )[0]
        health_factor = user_data[5] / 10**18
        health_factor = min(health_factor, 100)
        return (self.step, health_factor)
