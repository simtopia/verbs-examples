import numpy as np
import verbs


class Gbm:
    """
    Geometric brownian motion modelling the price of tokens A and B in USD.
    We assume that token B is some stablecoin so its price remains constant.
    """

    def __init__(
        self, mu: float, sigma: float, token_a_price: int, token_b_price: int, dt: float
    ):
        self.mu = mu
        self.sigma = sigma
        self.token_a_price = token_a_price
        self.token_b_price = token_b_price
        self.token_a_price_with_impact = token_a_price
        self.dt = dt

    def update(self, rng: np.random.Generator, price_impact: float):
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
        new_price_a_w_impact = new_price_a + price_impact

        # update price values
        self.token_a_price = new_price_a
        self.token_a_price_with_impact = new_price_a_w_impact

    def get_sqrt_price_token_a_x96(self):
        price = self.token_a_price_with_impact / self.token_b_price
        return np.sqrt(price) * 2**96

    def get_sqrt_price_token_b_x96(self):
        price = self.token_b_price / self.token_a_price_with_impact
        return np.sqrt(price) * 2**96

    def get_price_token_a(self):
        return self.token_a_price_with_impact / self.token_b_price


class UniswapAgent:
    """
    Agent that makes trades in Uniswap and the external market in order
    to make arbitrage.
    """

    def __init__(
        self,
        env,
        i: int,
        swap_router_abi,
        uniswap_pool_abi,
        fee: int,
        swap_router_address: bytes,
        uniswap_pool_address: bytes,
        # token A is considered to be the risky asset
        token_a_address: bytes,
        # token B is considered to be less risky / stablecoin
        token_b_address: bytes,
        mu: float,
        sigma: float,
        dt: float,
    ):
        self.address = verbs.utils.int_to_address(i)
        env.create_account(self.address, int(1e25))

        self.swap_router_abi = swap_router_abi
        self.uniswap_pool_abi = uniswap_pool_abi
        self.swap_router_address = swap_router_address
        self.uniswap_pool_address = uniswap_pool_address
        self.uniswap_fee = fee
        self.weth_address = token_a_address
        self.dai_address = token_b_address

        self.token_b = token_b_address  # stablecoin.
        self.token0_address = self.uniswap_pool_abi.token0.call(
            env, self.address, self.uniswap_pool_address, []
        )[0][0]
        self.token1_address = self.uniswap_pool_abi.token1.call(
            env, self.address, self.uniswap_pool_address, []
        )[0][0]
        self.fee = fee

        # external market model.
        # we initialise it at the same price as the Uniswap price
        # Uniswap returns price of token0 in terms of token1
        sqrt_price_uniswap_x96 = self.get_sqrt_price_x96_uniswap(env)

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
        # Variables to calculate price impact of Uniswap on the external exchange
        self.dt = dt
        self.beta = 2.0
        self.transient_impact = 0

        # step of simulator
        self.step = 0

    def get_sqrt_price_x96_uniswap(self, env) -> int:
        """get sqrt price from uniswap pool.
        Uniswap returns price of token0 in terms of token1
        """

        slot0 = self.uniswap_pool_abi.slot0.call(
            env, self.address, self.uniswap_pool_address, []
        )[0]
        sqrt_price_uniswap_x96 = slot0[0]
        return sqrt_price_uniswap_x96

    def get_swap_size_to_increase_uniswap_price(
        self,
        sqrt_price_external_market_x96: int,
        sqrt_price_uniswap_x96: int,
        liquidity: int,
    ):
        """
        Gets the swap parameters so that, after the swap, the price in Uniswap
        is the same as the price in the external market. We know that in
        Uniswap v2 (or v3 if there is not a tick range change), we have
        :math:`L = \\frac{\\Delta y}{\\Delta \\sqrt{P}}` where y is the
        numeraire (in our case the debt asset), and P is the price of the
        collateral in terms of the numeraire.
        Ref: https://atiselsts.github.io/pdfs/uniswap-v3-liquidity-math.pdf
        """
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
        """
        Gets the swap parameters so that, after the swap, the price in
        Uniswap is the same as the price in the external market. We
        know that in Uniswap v3 (or v2), we have
        :math:`L = \\frac{\\Delta y}{\\Delta \\sqrt{P}}` where y is
        the numeraire (in our case the debt asset), and P is the price
        of the collateral in terms of the numeraire.
        """
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

    def update(self, rng: np.random.Generator, env):
        # get sqrt price from uniswap pool. Uniswap returns price of
        # token0 in terms of token1
        sqrt_price_uniswap_x96 = self.get_sqrt_price_x96_uniswap(env)

        # We assume that trades on Uniswap have a price impact on the external
        # exchange. This is accumulated with an exponential decay
        if self.step > 0:
            current_price_impact = self.get_price_impact_in_external_market(env)
            self.transient_impact = (
                np.exp(-self.beta * self.dt) * self.transient_impact
                + current_price_impact
            )

        # get liquidity from uniswap pool
        liquidity = self.uniswap_pool_abi.liquidity.call(
            env, self.address, self.uniswap_pool_address, []
        )[0][0]

        # external market update
        self.external_market.update(rng, 0.1 * self.transient_impact)

        if self.token_b == self.token1_address:
            sqrt_price_external_market_x96 = (
                self.external_market.get_sqrt_price_token_a_x96()
            )
        else:
            sqrt_price_external_market_x96 = (
                self.external_market.get_sqrt_price_token_b_x96()
            )

        # Find encoded swap params so that price of uniswap after
        # swap matches the price of the external market
        # sqrt_price_external_market > sqrt_price_uniswap_x96,
        # the uniswap agent wants to buy collateral asset
        # (and sell debt asset) to increase the price of Uniswap
        # sqrt_price_external_market < sqrt_price_uniswap_x96,
        # the uniswap agent wants to sell collateral asset
        # (and buy debt asset) to decrease the price of Uniswap
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

    def record(self, env):
        # Get sqrt price from uniswap pool. Uniswap returns price of
        # token0 in terms of token1
        sqrt_price_uniswap_x96 = self.get_sqrt_price_x96_uniswap(env)

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

    def get_price_impact_in_external_market(self, env) -> float:
        """
        We assume that a trade in Uniswap has transient impact
        on the external exchange
        """
        sqrt_price_uniswap_x96 = self.get_sqrt_price_x96_uniswap(env)
        if self.token_b == self.token1_address:
            token_a_price_uniswap = (sqrt_price_uniswap_x96 / 2**96) ** 2
        else:
            token_a_price_uniswap = (2**96 / sqrt_price_uniswap_x96) ** 2
        token_a_price_external = self.external_market.get_price_token_a()
        return token_a_price_uniswap - token_a_price_external
