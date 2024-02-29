"""
Agent that trades token on uniswap to follow an external market
"""

import math
from typing import List, Tuple

import numpy as np
import verbs
from scipy.optimize import root_scalar

TICK_SPACING = {100: 1, 500: 10, 3000: 60, 10000: 200}


def tick_from_price(sqrt_price_x96: int, uniswap_fee: int) -> int:
    """
    Get tick from price and fee

    Parameters
    ----------
    sqrt_price_x96: int
        Square root of price times 2\ :sup:`96`
    uniswap_fee: int
        Uniswap fee. Possible values [100,500,3000,10000]

    Returns
    -------
    int
        Lower tick of input price
    """
    price = (sqrt_price_x96 / 2**96) ** 2
    tick = math.floor(math.log(price, 1.001))
    tick_lower = tick - (tick % TICK_SPACING[uniswap_fee])
    return tick_lower


def price_from_tick(tick: int) -> int:
    """
    Get price from tick

    Parameters
    ----------
    tick: int
        Lower tick of input price

    Returns
    -------
    int
        Square root of price times 2\ :sup:`96`
    """
    sqrt_price_x96 = np.sqrt(1.001**tick) * 2**96
    return sqrt_price_x96


class Gbm:
    """
    Geometric brownian motion modelling the price of two tokens

    Notes
    -----
    We assume that token B is some stablecoin so its price remains constant.
    """

    def __init__(
        self, mu: float, sigma: float, token_a_price: int, token_b_price: int, dt: float
    ):
        """
        Parameters
        ----------
        mu: float
            Drift of GBM
        sigma: float
            Volatility of GBM
        token_a_price: int
            Initial price of token A
        token_b_price: int
            Initial price of token B
        dt: float
            Time step of time discretisation for the SDE solver scheme
        """
        self.mu = mu
        self.sigma = sigma
        self.token_a_price = token_a_price
        self.token_b_price = token_b_price
        self.token_a_price_with_impact = token_a_price
        self.dt = dt

    def update(self, rng: np.random.Generator, price_impact: float):
        """
        Update Gbm

        Update GBM price using:

        * :math:`P^a_{t+dt} = P^a_t * exp((\\mu-0.5*\\sigma^2)dt +
          \sigma * (W_{t+dt} - W_{t}))`
        * :math:`P^{a, impact}_{t+dt} = P^a_{t+dt} + price_impact`
        * :math:`P^b` is constant

        Notes
        -----
        We consider an impact on the price of token A. This impact can be modelled
        in different ways, e.g. as a transient price impact given by the trades.

        Parameters
        ----------
        rng: np.random.Generator
            Numpy random generator, used for any random sampling
            to ensure determinism of the simulation.
        price_impact: float
            Network/EVM that the simulation interacts with.
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

    def get_sqrt_price_token_a_x96(self) -> float:
        """
        Get price of token A in terms of token B

        Notes
        -----
        We return the square root of the price times 2\ :sup:`96` for a fair comparison
        with the price values returned by the Uniswap contract.

        Returns
        -------
        float
            Square root of the price of token A in terms of token B times 2\ :sup:`96`
        """
        price = self.token_a_price_with_impact / self.token_b_price
        return np.sqrt(price) * 2**96

    def get_sqrt_price_token_b_x96(self):
        """
        Get price of token B in terms of token A

        Notes
        -----
        We return the square root of the price times 2\ :sup:`96`  for a fair comparison
        with the price values returned by the Uniswap contract.

        Returns
        -------
        float
            Square root of the price of token B in terms of token A times 2\ :sup:`96`
        """
        price = self.token_b_price / self.token_a_price_with_impact
        return np.sqrt(price) * 2**96

    def get_price_token_a(self):
        return self.token_a_price_with_impact / self.token_b_price


class BaseUniswapAgent:
    """
    Base agent that makes trades in Uniswap
    """

    def __init__(
        self,
        env,
        i: int,
        swap_router_abi,
        uniswap_pool_abi,
        quoter_abi,
        fee: int,
        swap_router_address: bytes,
        uniswap_pool_address: bytes,
        quoter_address: bytes,
        # token A is considered to be the risky asset
        token_a_address: bytes,
        # token B is considered to be less risky / stablecoin
        token_b_address: bytes,
    ):
        """
        Initialise the Uniswap agent and create the corresponding
        account in the EVM.

        The agent stores the ABIs of the Uniswap contracts
        and the token contracts that they will be interacting with.
        ABIs are previously loaded using the function :py:func:`verbs.abi.load_abi`.

        Parameters
        ----------
        env: verbs.types.Env
            Simulation environment
        i: int
            Agent index in the simulation
        swap_router_abi: type
            abi of the Uniswap v3 SwapRouter contract
        uniswap_pool_abi: type
            abi of the Uniswap v3 pool contract
        quoter_abi: type
            abi of the Uniswap v3 QuoterV2 contract
        fee: int
            Fee tier of the Uniswap v3 pool for the pair (token_a, token_b)
        swap_router_address: bytes
            Address of the SwapRouter contract
        uniswap_pool_address: bytes
            Address of Uniswap v3 pool for the pair (token_a, token_b)
        quoter_address: bytes
            Address of the QuoterV2 contract
        token_a_address: bytes
            Address of token_a
        token_b_address: bytes
            Address of token_b
        """
        self.address = verbs.utils.int_to_address(i)
        env.create_account(self.address, int(1e25))

        self.swap_router_abi = swap_router_abi
        self.uniswap_pool_abi = uniswap_pool_abi
        self.quoter_abi = quoter_abi
        self.swap_router_address = swap_router_address
        self.uniswap_pool_address = uniswap_pool_address
        self.quoter_address = quoter_address
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

    def get_sqrt_price_x96_uniswap(self, env) -> int:
        """
        Get sqrt price from uniswap pool

        Uniswap returns price of token0 in terms of token1

        Notes
        -----
        Uniswap sorts of token0 and token1 by their addresses.

        Parameters
        ----------
        env: verbs.types.Env
            Simulation environment

        Returns
        -------
        int
            Square root of the price times 2\ :sup:`96` of token0 in terms of token1
        """

        slot0 = self.uniswap_pool_abi.slot0.call(
            env, self.address, self.uniswap_pool_address, []
        )[0]
        sqrt_price_uniswap_x96 = slot0[0]
        return sqrt_price_uniswap_x96

    def get_swap_size_to_increase_uniswap_price(
        self,
        env,
        sqrt_target_price_x96: int,
        sqrt_price_uniswap_x96: int,
        liquidity: int,
        exact: bool = True,
    ) -> verbs.types.Transaction:
        """
        Get swap parameters to match target price

        Gets the swap parameters so that, after the swap, the price in Uniswap
        is the same as the target price. We know that in
        Uniswap v2 (or v3 if there is not a tick range change), we have
        :math:`L = \\frac{\\Delta y}{\\Delta \\sqrt{P}}` where y is the
        numeraire (in our case the debt asset), and P is the price of the
        collateral in terms of the numeraire.

        If there is a tick range and ``exact=True``, the agent performs
        an iterative calculation to find the right trade.

        References
        ----------
        #. https://atiselsts.github.io/pdfs/uniswap-v3-liquidity-math.pdf

        Parameters
        ----------
        env: verbs.types.Env
            Simulation environment
        sqrt_target_price_x96: int
            Sqrt of target price times 2\ :sup:`96`
        sqrt_price_uniswap_x96: int
            Sqrt of current uniswap price times 2\ :sup:`96`
        liquidity: int
            Liquidity of Uniswap in the current tick range
        exact: bool
            Boolean indicating whether to perform the iterative calculation
            to find the right trade.

        Returns
        -------
        verbs.types.Transaction
            Trade transaction
        """

        change_sqrt_price_x96 = sqrt_target_price_x96 - sqrt_price_uniswap_x96
        change_token_1 = int(liquidity * change_sqrt_price_x96 / 2**96)
        if change_token_1 == 0:
            return None

        def _quote_price(change_token_1):
            quote = self.quoter_abi.quoteExactInputSingle.call(
                env,
                self.address,
                self.quoter_address,
                [
                    (
                        self.token1_address,
                        self.token0_address,
                        int(change_token_1),
                        self.fee,
                        0,
                    )
                ],
            )[0]
            quoted_price = quote[1]
            return quoted_price

        if exact:
            # calculate the exact trade to match prices
            # this calculation will take into account
            # different liquidities in different tick ranges
            try:
                sol = root_scalar(
                    lambda x: _quote_price(x) - sqrt_target_price_x96,
                    x0=change_token_1,
                    method="newton",
                    maxiter=5,
                )
                change_token_1 = sol.root
            except:  # noqa: E722
                return None

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
                    int(change_token_1),
                    0,
                    0,
                )
            ],
        )
        return swap

    def get_swap_size_to_decrease_uniswap_price(
        self,
        env,
        sqrt_target_price_x96: int,
        sqrt_price_uniswap_x96: int,
        liquidity: int,
        exact: bool = True,
    ) -> verbs.types.Transaction:
        """
        Get swap parameters to match target price

        Gets the swap parameters so that, after the swap, the price in
        Uniswap is the same as the target price. We
        know that in Uniswap v3 (or v2), we have
        :math:`L = \\frac{\\Delta y}{\\Delta \\sqrt{P}}` where y is
        the numeraire (in our case the debt asset), and P is the price
        of the collateral in terms of the numeraire.

        If there is a tick range and ``exact=True``, the agent performs
        an iterative calculation to find the right trade.

        References
        ----------
        #. https://atiselsts.github.io/pdfs/uniswap-v3-liquidity-math.pdf

        Parameters
        ----------
        env: verbs.types.Env
            Simulation environment
        sqrt_target_price_x96: int
            Sqrt of target price times 2\ :sup:`96`
        sqrt_price_uniswap_x96: int
            Sqrt of current uniswap price times 2\ :sup:`96`
        liquidity: int
            Liquidity of Uniswap in the current tick range
        exact: bool
            Boolean indicating whether to perform the iterative calculation
            to find the right trade.

        Returns
        -------
        verbs.types.Transaction
            Trade transaction
        """

        change_sqrt_price_x96 = sqrt_price_uniswap_x96 - sqrt_target_price_x96
        change_token_1 = int(liquidity * change_sqrt_price_x96 / 2**96)
        if change_token_1 == 0:
            return None

        def _quote_price(change_token_1):
            quote = self.quoter_abi.quoteExactOutputSingle.call(
                env,
                self.address,
                self.quoter_address,
                [
                    (
                        self.token0_address,
                        self.token1_address,
                        int(change_token_1),
                        self.fee,
                        0,
                    )
                ],
            )[0]
            quoted_price = quote[1]
            return quoted_price

        if exact:
            # calculate the exact trade to match prices
            # this calculation will take into account
            # different liquidities in different tick ranges
            try:
                sol = root_scalar(
                    lambda x: _quote_price(x) - sqrt_target_price_x96,
                    method="newton",
                    x0=change_token_1,
                    maxiter=5,
                )
                change_token_1 = sol.root
            except:  # noqa: E722
                return None

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
                    int(change_token_1),
                    10**32,
                    0,
                )
            ],
        )
        return swap


class UniswapAgent(BaseUniswapAgent):
    """
    Agent that makes trades in Uniswap and the external market in order
    to make arbitrage
    """

    def __init__(
        self,
        env,
        i: int,
        swap_router_abi,
        uniswap_pool_abi,
        quoter_abi,
        fee: int,
        swap_router_address: bytes,
        uniswap_pool_address: bytes,
        quoter_address: bytes,
        # token A is considered to be the risky asset
        token_a_address: bytes,
        # token B is considered to be less risky / stablecoin
        token_b_address: bytes,
        mu: float,
        sigma: float,
        dt: float,
    ):
        """
        Initialise the Uniswap agent and create the account

        The agent stores the ABIs of the Uniswap contracts
        and the token contracts that they will be interacting with.
        ABIs are previously loaded using the function :py:func:`verbs.abi.load_abi`.

        The agent also has access to an external market, modelled by a Gbm,
        that is set as an attribute of the agents.

        Parameters
        ----------
        env: verbs.types.Env
            Simulation environment
        i: int
            Agent index in the simulation
        swap_router_abi: type
            abi of the Uniswap v3 SwapRouter contract
        uniswap_pool_abi: type
            abi of the Uniswap v3 pool contract
        quoter_abi: type
            abi of the Uniswap v3 QuoterV2 contract
        fee: int
            Fee tier of the Uniswap v3 pool for the pair (token_a, token_b)
        swap_router_address: bytes
            Address of the SwapRouter contract
        uniswap_pool_address: bytes
            Addres of Uniswap v3 pool for the pair (token_a, token_b)
        quoter_address: bytes
            Address of the QuoterV2 contract
        token_a_address: bytes
            Address of token_a
        token_b_address: bytes
            Address of token_b
        mu: float
            Drift of the Gbm
        sigma: float
            Volatility of the Gbm
        dt: float
            Time step of time discretisation for the Gbm solver.
        """
        super().__init__(
            env=env,
            i=i,
            swap_router_abi=swap_router_abi,
            uniswap_pool_abi=uniswap_pool_abi,
            quoter_abi=quoter_abi,
            swap_router_address=swap_router_address,
            uniswap_pool_address=uniswap_pool_address,
            quoter_address=quoter_address,
            fee=fee,
            token_a_address=token_a_address,
            token_b_address=token_b_address,
        )

        # External market model.
        # we initialise it at the same price as the Uniswap price
        # Uniswap returns price of token0 in terms of token1
        sqrt_price_uniswap_x96 = self.get_sqrt_price_x96_uniswap(env)

        if self.token_b == self.token1_address:
            self.init_token_a_price = (sqrt_price_uniswap_x96 / 2**96) ** 2
            token_b_price = 1
        else:
            self.init_token_a_price = (2**96 / sqrt_price_uniswap_x96) ** 2
            token_b_price = 1

        self.external_market = Gbm(
            mu=mu,
            sigma=sigma,
            token_a_price=self.init_token_a_price,
            token_b_price=token_b_price,
            dt=dt,
        )
        # Variables to calculate price impact of Uniswap on the external exchange
        self.dt = dt
        self.beta = 2.0
        self.transient_impact = 0

        # step of simulator
        self.step = 0

    def update(self, rng: np.random.Generator, env) -> List[verbs.types.Transaction]:
        """
        Update the state of the agent and returns
        list of transactions according to their policy.

        The Uniswap agent will

        * Check the price in the external market and in the Uniswap pool.
        * Calculate the trade to do in Uniswap in order to realize a profit.

        Parameters
        ----------
        rng: np.random.Generator
            Numpy random generator, used for any random sampling
            to ensure determinism of the simulation.
        env: verbs.types.Env
            Network/EVM that the simulation interacts with.

        Returns
        -------
        list
            List of transactions to be processed in the next block
            of the simulation. This can be an empty list if the
            agent is not submitting any transactions.

        """
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
                env=env,
                sqrt_target_price_x96=sqrt_price_external_market_x96,
                sqrt_price_uniswap_x96=sqrt_price_uniswap_x96,
                liquidity=liquidity,
            )
        else:
            swap_call = self.get_swap_size_to_decrease_uniswap_price(
                env=env,
                sqrt_target_price_x96=sqrt_price_external_market_x96,
                sqrt_price_uniswap_x96=sqrt_price_uniswap_x96,
                liquidity=liquidity,
            )
        self.step += 1

        if swap_call is not None:
            return [swap_call]
        else:
            return []

    def record(self, env) -> Tuple[float, float]:
        """
        Record the state of the agent

        This method is called at the end of each step for all agents.
        It should return any data to be recorded over the course
        of the simulation.

        Parameters
        ----------
        env: verbs.types.Env
            Network/EVM that the simulation interacts with.

        Returns
        -------
        tuple[float, float]
            Tuple containing:
            - Price in Uniswap of token0 in terms of token1
            - Price in the external market of token0 in terms of token1
        """
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
        Estimate Uniswap trade impact on the external market

        We assume that a trade in Uniswap has transient impact
        on the external exchange.

        Parameters
        ----------
        env: verbs.types.Env
            Network/EVM that the simulation interacts with.

        Returns
        -------
        float
            Transient impact
        """
        sqrt_price_uniswap_x96 = self.get_sqrt_price_x96_uniswap(env)
        if self.token_b == self.token1_address:
            token_a_price_uniswap = (sqrt_price_uniswap_x96 / 2**96) ** 2
        else:
            token_a_price_uniswap = (2**96 / sqrt_price_uniswap_x96) ** 2
        token_a_price_external = self.external_market.get_price_token_a()
        return token_a_price_uniswap - token_a_price_external


class DummyUniswapAgent(UniswapAgent):
    """
    Dummy uniswap agent used for cache generation

    Uniswap agent that queries the EVM database
    for a wide range of Uniswap price ticks.
    Useful to initialise the cache of a simulation
    """

    def __init__(
        self,
        env,
        i: int,
        swap_router_abi,
        uniswap_pool_abi,
        quoter_abi,
        fee: int,
        swap_router_address: bytes,
        uniswap_pool_address: bytes,
        quoter_address: bytes,
        # token A is considered to be the risky asset
        token_a_address: bytes,
        # token B is considered to be less risky / stablecoin
        token_b_address: bytes,
        mu: float,
        sigma: float,
        dt: float,
        sim_n_steps: int,
    ):
        """
        Initialise the Uniswap agent and create the corresponding
        account in the EVM.

        The agent stores the ABIs of the Uniswap contracts
        and the token contracts that they will be interacting with.
        ABIs are previously loaded using the function :py:func:`verbs.abi.load_abi`.

        The agent also has access to an external market, modelled by a Gbm,
        that is set as an attribute of the agents.

        Notes
        -----
        This agent should only be used in a simulation to initialise the Cache
        of the EVM database. The drift and the volatility of the external market
        are artificially calibrated in order for the agent to explore a wide range
        of Uniswap price ticks and thus find out the right storage slots to be
        saved in the Cache.

        Parameters
        ----------
        env: verbs.types.Env
            Simulation environment
        i: int
            Agent index in the simulation
        swap_router_abi: type
            abi of the Uniswap v3 SwapRouter contract
        uniswap_pool_abi: type
            abi of the Uniswap v3 pool contract
        quoter_abi: type
            abi of the Uniswap v3 QuoterV2 contract
        fee: int
            Fee tier of the Uniswap v3 pool for the pair (token_a, token_b)
        swap_router_address: bytes
            Address of the SwapRouter contract
        uniswap_pool_address: bytes
            Addres of Uniswap v3 pool for the pair (token_a, token_b)
        quoter_address: bytes
            Address of the QuoterV2 contract
        token_a_address: bytes
            Address of token_a
        token_b_address: bytes
            Address of token_b
        mu: float
            Drift of the Gbm
        sigma: float
            Volatility of the Gbm
        dt: float
            Time step of time discretisation for the Gbm solver.
        """
        # Calibrate mu and sigma in order to explore Uniswap pool
        # storage values for simulation
        super().__init__(
            env=env,
            i=i,
            swap_router_abi=swap_router_abi,
            uniswap_pool_abi=uniswap_pool_abi,
            quoter_abi=quoter_abi,
            swap_router_address=swap_router_address,
            uniswap_pool_address=uniswap_pool_address,
            quoter_address=quoter_address,
            fee=fee,
            token_a_address=token_a_address,
            token_b_address=token_b_address,
            mu=mu,
            sigma=0,
            dt=dt,
        )
        self.sim_n_steps = sim_n_steps

        # calibrate mu to explore the pool
        upper_bound_price = 1.7 * self.init_token_a_price
        lower_bound_price = 0.3 * self.init_token_a_price

        self.mu0 = (
            1
            / (dt * float(sim_n_steps // 3))
            * np.log(upper_bound_price / self.init_token_a_price)
        )
        self.mu1 = (
            1
            / (dt * sim_n_steps - dt * float(sim_n_steps // 3))
            * np.log(lower_bound_price / upper_bound_price)
        )

    def update(self, rng: np.random.Generator, env) -> List[verbs.types.Transaction]:
        """
        Update the state of the agent

        Makes an exploratory update by manually changing
        the drift of the external market.

        Parameters
        ----------
        rng: np.random.Generator
            Numpy random generator, used for any random sampling
            to ensure determinism of the simulation.
        env: verbs.types.Env
            Network/EVM that the simulation interacts with.

        Returns
        -------
        list
            List of transactions to be processed in the next block
            of the simulation. This can be an empty list if the
            agent is not submitting any transactions.
        """
        if self.step < self.sim_n_steps // 3:
            self.external_market.mu = self.mu0
        else:
            self.external_market.mu = self.mu1
        tx = super().update(rng, env)
        return tx
