"""
Agent that monitors Aave borrower positions and liquidates them
"""
from typing import List, Tuple

import numpy as np
import verbs


class LiquidationAgent:
    """
    Agent that monitors Aave borrowers and liquidate positions
    """

    def __init__(
        self,
        env,
        i: int,
        pool_implementation_abi: type,
        mintable_erc20_abi: type,
        pool_address: bytes,
        token_a_address: bytes,
        token_b_address: bytes,
        liquidation_addresses: List[bytes],
        uniswap_pool_abi: type,
        quoter_abi: type,
        swap_router_abi: type,
        uniswap_pool_address: bytes,
        quoter_address: bytes,
        swap_router_address: bytes,
        uniswap_fee: int,
    ):
        """
        Initialise the Liquidator agent and create the corresponding
        account in the EVM.

        The agent stores the ABIs of the Aave contracts, the Uniswap contracts
        and the token contracts that they will be interacting with.
        ABIs are previously loaded using the function :py:func:`verbs.abi.load_abi`.

        Parameters
        ----------
        env: verbs.types.Env
            Simulation environment
        i: int
            Agent index in the simulation
        pool_implementation_abi: type
            abi of the Aave v3 pool contract
        mintable_erc20_abi: type
            abi of ERC20 contract
        pool_address: bytes
            Addres of Aave v3 pool contract
        token_a_address: bytes
            Address of collateral token (usually the risky token)
        token_b_address: bytes
            Address of debt token (usually the less risky token)
        liquidation_addresses: list[bytes]
            List of borrowers' addresses that the liquidator will be monitoring.
        uniswap_pool_abi: type
            abi of the Uniswap v3 pool contract
        quoter_abi: type
            abi of the Uniswap v3 QuoterV2 contract
        swap_router_abi: type
            abi of the Uniswap v3 SwapRouter contract
        uniswap_pool_address: bytes
            Addres of Uniswap v3 pool for the pair (token_a, token_b)
        quoter_address: bytes
            Address of the QuoterV2 contract
        swap_router_address: bytes
            Address of the SwapRouter contract
        uniswap_fee: int
            Fee tier of the Uniswap v3 pool for the pair (token_a, token_b)
        """
        self.address = verbs.utils.int_to_address(i)
        env.create_account(self.address, int(1e35))

        # Aave
        self.pool_implementation_abi = pool_implementation_abi
        self.pool_address = pool_address
        self.liquidation_addresses = liquidation_addresses

        # Tokens
        # collateral token - risky asset
        self.token_a_address = token_a_address
        # Debt token - stablecoin
        self.token_b_address = token_b_address

        self.decimals_token_b = mintable_erc20_abi.decimals.call(
            env, self.address, self.token_b_address, []
        )[0][0]
        self.mintable_erc20_abi = mintable_erc20_abi

        # Uniswap
        self.uniswap_pool_abi = uniswap_pool_abi
        self.quoter_abi = quoter_abi
        self.swap_router_abi = swap_router_abi

        self.uniswap_pool_address = uniswap_pool_address
        self.quoter_address = quoter_address
        self.swap_router_address = swap_router_address

        self.uniswap_fee = uniswap_fee

        # Liquidator's wallet
        self.balance_debt_asset = []
        self.balance_collateral_asset = []

        # simulation steps
        self.step = 0

    def accountability(self, env, liquidation_address, amount: int) -> bool:
        """
        Calculates if a liquidation is profitable

        Makes the accountability of a liquidation and returns a boolean indicating
        whether the liquidation is profitable or not

        Parameters
        ----------
        env: verbs.types.Env
            Simulation environment.
        liquidation_address: bytes
            Liquidation address for which the Liquidator calculates the profitability
            of the liquidation.
        amount: int
            Amount to be liquidated

        Returns
        -------
        bool
            ``True`` if the liquidation is profitable
        """

        try:
            liquidation_call_event = self.pool_implementation_abi.liquidationCall.call(
                env,
                self.address,
                self.pool_address,
                [
                    self.token_a_address,
                    self.token_b_address,
                    liquidation_address,
                    amount,
                    True,
                ],
            )[1]
        except verbs.envs.RevertError:
            return False

        decoded_liquidation_call_event = (
            self.pool_implementation_abi.LiquidationCall.decode(
                liquidation_call_event[-1][1]
            )
        )

        debt_to_cover = decoded_liquidation_call_event[0]
        liquidated_collateral_amount = decoded_liquidation_call_event[1]

        quote = self.quoter_abi.quoteExactOutputSingle.call(
            env,
            self.address,
            self.quoter_address,
            [
                (
                    self.token_a_address,
                    self.token_b_address,
                    debt_to_cover,
                    self.uniswap_fee,
                    0,
                )
            ],
        )[0]

        amount_collateral_from_swap = quote[0]
        return amount_collateral_from_swap < liquidated_collateral_amount

    def update(self, rng: np.random.Generator, env) -> List[verbs.types.Transaction]:
        """
        Update the state of the agent and returns
        list of transactions according to their policy.

        The liquidator agent will

        * Liquidate positions in Aave that are in distress
        * Realize a profit on Uniswap by selling the collateral
          obtained from liquidations

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
            agent is not submitting any transacti
        """
        current_balance_collateral_asset = self.mintable_erc20_abi.balanceOf.call(
            env,
            self.address,
            self.token_a_address,
            [
                self.address,
            ],
        )[0][0]
        self.balance_collateral_asset.append(current_balance_collateral_asset)
        current_balance_debt_asset = self.mintable_erc20_abi.balanceOf.call(
            env,
            self.address,
            self.token_b_address,
            [
                self.address,
            ],
        )[0][0]
        self.balance_debt_asset.append(current_balance_debt_asset)

        # get the users'data
        users_data = []
        for borrower in self.liquidation_addresses:
            borrower_data = self.pool_implementation_abi.getUserAccountData.call(
                env, self.address, self.pool_address, [borrower]
            )[0]
            users_data.append((borrower, borrower_data))

        # filter risky positions
        risky_positions = filter(lambda x: x[1][5] < 10**18, users_data)

        # filter those positions for which liquidating is profitable
        # Note: https://docs.aave.com/developers/core-contracts/pool#liquidationcall
        # debtToCover parameter can be set to uint(-1) and the protocol will proceed
        # with the highest possible liquidation allowed by the close factor.
        liquidatable_positions = filter(
            lambda x: self.accountability(env, x[0], 10**32), risky_positions
        )

        # create transactions
        tx = []
        for position in liquidatable_positions:
            tx.append(
                self.pool_implementation_abi.liquidationCall.transaction(
                    self.address,
                    self.pool_address,
                    [
                        self.token_a_address,
                        self.token_b_address,
                        position[0],
                        10**32,
                        False,
                    ],
                    checked=False,
                )
            )

        if self.step > 0:
            debt = int(self.balance_debt_asset[-2] - self.balance_debt_asset[-1])
            # check if liquidator has open short position in the debt asset
            if debt > 0:
                swap_tx = self.swap_router_abi.exactOutputSingle.transaction(
                    self.address,
                    self.swap_router_address,
                    [
                        (
                            self.token_a_address,
                            self.token_b_address,
                            self.uniswap_fee,
                            self.address,
                            10**32,
                            debt,
                            current_balance_collateral_asset,
                            0,
                        )
                    ],
                )
                tx.append(swap_tx)

        # sim step
        self.step += 1

        return tx

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
            - Balance of collateral asset in the current step.
            - Balance of debt asset in the current step.
        """
        current_balance_collateral_asset = self.mintable_erc20_abi.balanceOf.call(
            env,
            self.address,
            self.token_a_address,
            [
                self.address,
            ],
        )[0][0]
        current_balance_debt_asset = self.mintable_erc20_abi.balanceOf.call(
            env,
            self.address,
            self.token_b_address,
            [
                self.address,
            ],
        )[0][0]

        return (
            current_balance_collateral_asset / 10**18,
            current_balance_debt_asset / 10**18,
        )


class AdversarialLiquidationAgent(LiquidationAgent):
    """
    Liquidation agent that manipulates the price in Uniswap
    to bring borrowers positions into distress
    """

    def __init__(
        self,
        env,
        i: int,
        pool_implementation_abi: type,
        mintable_erc20_abi: type,
        pool_address: bytes,
        token_a_address: bytes,
        token_b_address: bytes,
        liquidation_addresses: List,
        uniswap_pool_abi: type,
        quoter_abi: type,
        swap_router_abi,
        uniswap_pool_address: bytes,
        quoter_address: bytes,
        swap_router_address: bytes,
        uniswap_fee: int,
        aave_oracle_abi: type,
        aave_oracle_address: bytes,
    ):
        """
        Initialise the Liquidator agent and create the corresponding
        account in the EVM.

        The agent stores the ABIs of the Aave contracts, the Uniswap contracts
        and the token contracts that they will be interacting with.
        ABIs are previously loaded using the function :py:func:`verbs.abi.load_abi`.

        Parameters
        ----------
        env: verbs.types.Env
            Simulation environment
        i: int
            Agent index in the simulation
        pool_implementation_abi: type
            abi of the Aave v3 pool contract
        mintable_erc20_abi: type
            abi of ERC20 contract
        pool_address: bytes
            Addres of Aave v3 pool contract
        token_a_address: bytes
            Address of collateral token (usually the risky token)
        token_b_address: bytes
            Address of debt token (usually the less risky token)
        liquidation_addresses: list[bytes]
            List of borrowers' addresses that the liquidator will be monitoring.
        uniswap_pool_abi: type
            abi of the Uniswap v3 pool contract
        quoter_abi: type
            abi of the Uniswap v3 QuoterV2 contract
        swap_router_abi: type
            abi of the Uniswap v3 SwapRouter contract
        uniswap_pool_address: bytes
            Addres of Uniswap v3 pool for the pair (token_a, token_b)
        quoter_address: bytes
            Address of the QuoterV2 contract
        swap_router_address: bytes
            Address of the SwapRouter contract
        uniswap_fee: int
            Fee tier of the Uniswap v3 pool for the pair (token_a, token_b)
        aave_oracle_abi: type
            abi of the Aave oracle contract for the pair (token_a, token_b)
        aave_oracle_address: bytes
            Address of the Aave oracle contract for the pair (token_a, token_b)
        """
        super().__init__(
            env,
            i,
            pool_implementation_abi,
            mintable_erc20_abi,
            pool_address,
            token_a_address,
            token_b_address,
            liquidation_addresses,
            uniswap_pool_abi,
            quoter_abi,
            swap_router_abi,
            uniswap_pool_address,
            quoter_address,
            swap_router_address,
            uniswap_fee,
        )

        # Aave oracle
        self.aave_oracle_abi = aave_oracle_abi
        self.aave_oracle_address = aave_oracle_address

        # Uniswap token 0 and token 1
        self.token0_address = self.uniswap_pool_abi.token0.call(
            env, self.address, self.uniswap_pool_address, []
        )[0][0]
        self.token1_address = self.uniswap_pool_abi.token1.call(
            env, self.address, self.uniswap_pool_address, []
        )[0][0]

    def accountability(self, env, liquidation_address, amount: int) -> bool:
        """
        Calculates if a liquidation is profitable

        Makes the accountability of a liquidation and returns a boolean indicating
        whether the liquidation is profitable or not

        Parameters
        ----------
        env: verbs.types.Env
            Simulation environment.
        liquidation_address: bytes
            Liquidation address for which the Liquidator calculates the profitability
            of the liquidation.
        amount: int
            Amount to be liquidated

        Returns
        -------
        bool
            ``True`` if the liquidation is profitable.
        """
        if self.balance_debt_asset[-2] < self.balance_debt_asset[-1]:
            # The agents is long on the debt asset.
            # That means they have done a front-run trade
            # in order to make a liquidation
            return True
        else:
            return super().accountability(env, liquidation_address, amount)

    def update(self, rng: np.random.Generator, env):
        """
        Update the state of the agent and returns
        list of transactions according to their policy.

        The liquidator agent will

        * Monitor those positions in Aave that are close to being
          in distress, and check whether it would be profitable
          to make a trade in Uniswap to decrease the
          price of collateral in order to trigger liquidations.
        * Liquidate positions in Aave that are in distress.
        * Realize a profit on Uniswap by selling the collateral
          obtained from liquidations.

        References
        ----------

        #. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4540333

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
        # liquidation transactions + closing short collateral position on Uniswap
        tx = super().update(rng, env)

        # front-run trades on Uniswap
        debt_asset_price, _ = self.aave_oracle_abi.getAssetsPrices.call(
            env,
            self.address,
            self.aave_oracle_address,
            [[self.token_b_address, self.token_a_address]],
        )[0][0]
        # get price of Uniswap
        sqrt_price_x96 = self.uniswap_pool_abi.slot0.call(
            env, self.address, self.uniswap_pool_address, []
        )[0][0]
        total_debt_to_cover = 0
        for borrower in self.liquidation_addresses:
            # We calculate the upper bound of HF so that adversarial liquidation
            # is profitable
            # See https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4540333
            borrower_data = self.pool_implementation_abi.getUserAccountData.call(
                env, self.address, self.pool_address, [borrower]
            )[0]
            hf = borrower_data[5]

            # debtBase and aave oracle price have the same number of decimals (8)
            # so we do not need to re-scale anything.
            debt_to_cover = (
                borrower_data[1] * 10**self.decimals_token_b / (2 * debt_asset_price)
            )
            if debt_to_cover > 0:
                quote = self.quoter_abi.quoteExactOutputSingle.call(
                    env,
                    self.address,
                    self.quoter_address,
                    [
                        (
                            self.token_a_address,
                            self.token_b_address,
                            int(debt_to_cover),
                            self.uniswap_fee,
                            0,
                        )
                    ],
                )[0]
                sqrt_price_x96_after = quote[1]
                sqrt_upper_bound_hf = (
                    sqrt_price_x96 / sqrt_price_x96_after
                    if self.token_b_address == self.token1_address
                    else sqrt_price_x96_after / sqrt_price_x96
                )
                upper_bound_hf = sqrt_upper_bound_hf**2
                if 1.0 < hf / 10**18 and hf / 10**18 < upper_bound_hf:
                    total_debt_to_cover += debt_to_cover

        # front-run transaction
        if int(total_debt_to_cover) > 0:
            swap_tx = self.swap_router_abi.exactOutputSingle.transaction(
                self.address,
                self.swap_router_address,
                [
                    (
                        self.token_a_address,
                        self.token_b_address,
                        self.uniswap_fee,
                        self.address,
                        10**32,
                        int(total_debt_to_cover),
                        self.balance_collateral_asset[-1],
                        0,
                    )
                ],
            )
            tx.append(swap_tx)
        return tx
