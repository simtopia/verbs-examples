import typing

import numpy as np
import verbs


class LiquidationAgent:
    def __init__(
        self,
        network,
        i: int,
        pool_implementation_abi,
        mintable_erc20_abi,
        pool_address: str,
        token_a_address: str,
        token_b_address: str,
        liquidation_addresses: typing.List,
        uniswap_pool_abi,
        quoter_abi,
        swap_router_abi,
        uniswap_pool_address: str,
        quoter_address: str,
        swap_router_address: str,
        uniswap_fee: int,
    ):
        self.net = network
        self.address = verbs.utils.int_to_address(i)
        self.net.create_account(self.address, int(1e25))

        # Aave
        self.pool_implementation_abi = pool_implementation_abi
        self.pool_address = verbs.utils.hex_to_bytes(pool_address)
        self.liquidation_addresses = liquidation_addresses

        # tokens
        self.token_a_address = verbs.utils.hex_to_bytes(
            token_a_address
        )  # collateral token - risky asset
        self.token_b_address = verbs.utils.hex_to_bytes(
            token_b_address
        )  # debt token - stablecoin

        self.decimals_token_b = mintable_erc20_abi.decimals.call(
            self.net, self.address, self.token_b_address, []
        )[0][0]
        self.mintable_erc20_abi = mintable_erc20_abi

        # Uniswap
        self.uniswap_pool_abi = uniswap_pool_abi
        self.quoter_abi = quoter_abi
        self.swap_router_abi = swap_router_abi

        self.uniswap_pool_address = verbs.utils.hex_to_bytes(uniswap_pool_address)
        self.quoter_address = verbs.utils.hex_to_bytes(quoter_address)
        self.swap_router_address = verbs.utils.hex_to_bytes(swap_router_address)

        self.uniswap_fee = uniswap_fee

        # Liquidator's wallet
        self.balance_debt_asset = []
        self.balance_collateral_asset = []

        # simulation steps
        self.step = 0

    def accountability(self, liquidation_address, amount: int) -> bool:
        """Makes the accountability of a liquidation and returns a boolean indicating
        whether the liquidation is profitable or not
        """

        liquidation_call_event = self.pool_implementation_abi.liquidationCall.call(
            self.net,
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

        if liquidation_call_event:
            decoded_liquidation_call_event = (
                self.pool_implementation_abi.LiquidationCall.decode(
                    liquidation_call_event[-1][1]
                )
            )

            debt_to_cover = decoded_liquidation_call_event[0]
            liquidated_collateral_amount = decoded_liquidation_call_event[1]

            quote = self.quoter_abi.quoteExactOutputSingle.call(
                self.net,
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

    def update(self, rng: np.random.Generator, *args):
        current_balance_collateral_asset = self.mintable_erc20_abi.balanceOf.call(
            self.net,
            self.address,
            self.token_a_address,
            [
                self.address,
            ],
        )[0][0]
        current_balance_debt_asset = self.mintable_erc20_abi.balanceOf.call(
            self.net,
            self.address,
            self.token_b_address,
            [
                self.address,
            ],
        )[0][0]

        # get the users'data
        users_data = []
        for borrower in self.liquidation_addresses:
            borrower_data = self.pool_implementation_abi.getUserAccountData.call(
                self.net, self.address, self.pool_address, [borrower]
            )[0]
            users_data.append((borrower, borrower_data))

        # filter risky positions
        risky_positions = filter(lambda x: x[1][5] < 10**18, users_data)

        # filter thoses positions for which liquidating is profitable
        # Note: https://docs.aave.com/developers/core-contracts/pool#liquidationcall
        # debtToCover parameter can be set to uint(-1) and the protocol will proceed with the highest possible liquidation allowed by the close factor.
        liquidatable_positions = filter(
            lambda x: self.accountability(x[0], 10**32), risky_positions
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
                )
            )

        if self.step > 0:
            # check if liquidator has open short position in the debt asset
            if self.balance_debt_asset[-1] > current_balance_debt_asset:
                debt = self.balance_debt_asset[-1] - current_balance_debt_asset
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

        # update wallet
        self.balance_collateral_asset.append(current_balance_collateral_asset)
        self.balance_debt_asset.append(current_balance_debt_asset)
        # sim step
        self.step += 1

        return tx

    def record(
        self,
    ):
        current_balance_collateral_asset = self.mintable_erc20_abi.balanceOf.call(
            self.net,
            self.address,
            self.token_a_address,
            [
                self.address,
            ],
        )[0][0]
        current_balance_debt_asset = self.mintable_erc20_abi.balanceOf.call(
            self.net,
            self.address,
            self.token_b_address,
            [
                self.address,
            ],
        )[0][0]

        return (current_balance_collateral_asset, current_balance_debt_asset)
