"""
Agent that monitors Aave borrower positions and liquidates them
"""
import typing

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
        pool_implementation_abi,
        mintable_erc20_abi,
        pool_address: bytes,
        token_a_address: bytes,
        token_b_address: bytes,
        liquidation_addresses: typing.List,
        uniswap_pool_abi,
        quoter_abi,
        swap_router_abi,
        uniswap_pool_address: bytes,
        quoter_address: bytes,
        swap_router_address: bytes,
        uniswap_fee: int,
    ):
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

    def update(self, rng: np.random.Generator, env):
        """
        Update the state of the agent
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

    def record(self, env):
        """
        Record the state of the agent
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
    Liquidation agent that manipulates borrower positions via Uniswap
    """

    def __init__(
        self,
        env,
        i: int,
        pool_implementation_abi,
        mintable_erc20_abi,
        pool_address: bytes,
        token_a_address: bytes,
        token_b_address: bytes,
        liquidation_addresses: typing.List,
        uniswap_pool_abi,
        quoter_abi,
        swap_router_abi,
        uniswap_pool_address: bytes,
        quoter_address: bytes,
        swap_router_address: bytes,
        uniswap_fee: int,
        aave_oracle_abi,
        aave_oracle_address: bytes,
    ):
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
        Update the state of the agent
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
