import numpy as np
import verbs


class Gbm:
    def __init__(
        self, mu: float, sigma: float, token_a_price: int, token_b_price: int, dt: float
    ):
        self.mu = mu
        self.sigma = sigma
        self.token_a_price = token_a_price
        self.token_b_price = token_b_price
        self.dt = dt

    def update(
        self,
        rng: np.random.Generator,
    ):
        """
        Update Gbm:
        - P^a_{t+dt} = P^a_t * exp((mu-0.5*sigma^2)dt + sigma * (W_{t+dt} - W_{t}))
        - P^b is constant

        """
        z = rng.normal()
        new_price_a = self.token_a_price * np.exp(
            (self.mu - 0.5 * self.sigma**2) * self.dt
            + self.sigma * np.sqrt(self.dt) * z
        )

        # update price values
        self.token_a_price = int(new_price_a)

    def get_sqrt_price_token_a_x96(self):
        price = self.token_a_price / self.token_b_price
        return np.sqrt(price) * 2**96

    def get_sqrt_price_token_b_x96(self):
        price = self.token_b_price / self.token_a_price
        return np.sqrt(price) * 2**96

    def get_price_token_a(self):
        price = self.token_a_price / self.token_b_price
        return price


class UniswapAgent:
    def __init__(
        self,
        network,
        i: int,
        swap_router_abi,
        uniswap_pool_abi,
        fee: int,
        swap_router_address: str,
        uniswap_pool_address: str,
        token_a_address: str,  # external market (gbm) returns the price of token a in terms of token b
        token_b_address: str,  # token b is the numeraire
        token_a_price: int,
        token_b_price: int,
        mu: float,
        sigma: float,
        dt: float,
    ):
        self.net = network
        self.address = verbs.utils.int_to_address(i)
        self.net.create_account(self.address, int(1e25))
        self.swap_router_abi = swap_router_abi
        self.uniswap_pool_abi = uniswap_pool_abi
        self.swap_router_address = verbs.utils.hex_to_bytes(swap_router_address)
        self.uniswap_pool_address = verbs.utils.hex_to_bytes(uniswap_pool_address)
        self.uniswap_fee = fee
        self.weth_address = token_a_address
        self.dai_address = token_b_address

        self.token_b = token_b_address  # stablecoin.
        self.token0_address = self.uniswap_pool_abi.token0.call(
            self.net, self.net.admin_address, self.uniswap_pool_address, []
        )[0][0]
        self.token1_address = self.uniswap_pool_abi.token1.call(
            self.net, self.net.admin_address, self.uniswap_pool_address, []
        )[0][0]
        self.fee = fee

        # external market model.
        # we initialise it at the same price as the Uniswap price
        # Uniswap returns price of token0 in terms of token1
        slot0 = self.uniswap_pool_abi.slot0.call(
            self.net, self.net.admin_address, self.uniswap_pool_address, []
        )[0]
        sqrt_price_uniswap_x96 = slot0[0]

        if self.token_b == self.token1_address:
            token_a_price = (sqrt_price_uniswap_x96 / 2**96) ** 2
            token_b_price = 1
        else:
            token_a_price = (2**96 / sqrt_price_uniswap_x96) ** 2
            token_b_price = 1

        self.external_market = Gbm(
            mu=mu,
            sigma=sigma,
            token_a_price=token_a_price,
            token_b_price=token_b_price,
            dt=dt,
        )

        # step of simulator
        self.step = 0

    def get_swap_size_to_increase_uniswap_price(
        self,
        sqrt_price_external_market_x96: int,
        sqrt_price_uniswap_x96: int,
        liquidity: int,
    ):
        change_sqrt_price_x96 = sqrt_price_external_market_x96 - sqrt_price_uniswap_x96
        change_token_1 = int(liquidity * change_sqrt_price_x96 / 2**96)
        if change_token_1 > 0:
            swap = self.swap_router_abi.exactInputSingle.transaction(
                self.address,
                self.swap_router_address,
                [
                    (
                        self.token1_address,
                        self.token0_address,
                        self.fee,
                        self.address,
                        10**32,
                        change_token_1,
                        0,
                        0,
                    )
                ],
            )
            return swap
        else:
            return None

    def get_swap_size_to_decrease_uniswap_price(
        self,
        sqrt_price_external_market_x96: int,
        sqrt_price_uniswap_x96: int,
        liquidity: int,
    ):
        change_sqrt_price_x96 = sqrt_price_uniswap_x96 - sqrt_price_external_market_x96
        change_token_1 = int(liquidity * change_sqrt_price_x96 / 2**96)
        if change_token_1 > 0:
            swap = self.swap_router_abi.exactOutputSingle.transaction(
                self.address,
                self.swap_router_address,
                [
                    (
                        self.token0_address,
                        self.token1_address,
                        self.fee,
                        self.address,
                        10**32,
                        change_token_1,
                        10**32,
                        0,
                    )
                ],
            )
            return swap
        else:
            return None

    def update(self, rng: np.random.Generator, *args):
        # get sqrt price from uniswap pool. Uniswap returns price of token0 in terms of token1
        slot0 = self.uniswap_pool_abi.slot0.call(
            self.net, self.address, self.uniswap_pool_address, []
        )[0]
        sqrt_price_uniswap_x96 = slot0[0]

        # get liquidity from uniswap pool
        liquidity = self.uniswap_pool_abi.liquidity.call(
            self.net, self.address, self.uniswap_pool_address, []
        )[0][0]

        # external market update
        self.external_market.update(rng)

        if self.token_b == self.token1_address:
            sqrt_price_external_market_x96 = (
                self.external_market.get_sqrt_price_token_a_x96()
            )
        else:
            sqrt_price_external_market_x96 = (
                self.external_market.get_sqrt_price_token_b_x96()
            )

        # find encoded swap params so that price of uniswap after swap matches the price of the external market
        # sqrt_price_external_market > sqrt_price_uniswap_x96, the uniswap agent wants to buy collateral asset (and sell debt asset) to increase the price of Uniswap
        # sqrt_price_external_market < sqrt_price_uniswap_x96, the uniswap agent wants to sell collateral asset (and buy debt asset) to decrease the price of Uniswap
        if sqrt_price_external_market_x96 > sqrt_price_uniswap_x96:
            swap_call = self.get_swap_size_to_increase_uniswap_price(
                sqrt_price_external_market_x96=sqrt_price_external_market_x96,
                sqrt_price_uniswap_x96=sqrt_price_uniswap_x96,
                liquidity=liquidity,
            )
        else:
            swap_call = self.get_swap_size_to_decrease_uniswap_price(
                sqrt_price_external_market_x96=sqrt_price_external_market_x96,
                sqrt_price_uniswap_x96=sqrt_price_uniswap_x96,
                liquidity=liquidity,
            )
        self.step += 1

        if swap_call is not None:
            return [swap_call]
        else:
            return []

    def record(self):
        # get sqrt price from uniswap pool. Uniswap returns price of token0 in terms of token1
        slot0 = self.uniswap_pool_abi.slot0.call(
            self.net, self.net.admin_address, self.uniswap_pool_address, []
        )[0]
        sqrt_price_uniswap_x96 = slot0[0]

        if self.token_b == self.token1_address:
            sqrt_price_external_market_x96 = (
                self.external_market.get_sqrt_price_token_a_x96()
            )
        else:
            sqrt_price_external_market_x96 = (
                self.external_market.get_sqrt_price_token_b_x96()
            )

        sqrt_price_uniswap = sqrt_price_uniswap_x96 / 2**96
        sqrt_price_external_market = sqrt_price_external_market_x96 / (2**96)

        return (sqrt_price_uniswap**2, sqrt_price_external_market**2)
