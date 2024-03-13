"""
Agent that supplies and borrows tokens from an Aave pool
"""

from typing import List, Tuple

import numpy as np
import verbs


class BorrowAgent:
    """
    Borrower agent who supplies and borrows tokens from an Aave pool
    """

    def __init__(
        self,
        env,
        i: int,
        pool_implementation_abi: type,
        oracle_abi: type,
        mintable_erc20_abi: type,
        pool_address: bytes,
        oracle_address: bytes,
        token_a_address: bytes,
        token_b_address: bytes,
        activation_rate: float,
    ):
        """
        Initialise the Borrower agent and create the corresponding
        account in the EVM.

        The agent stores the ABIs of the Aave contracts and the token
        contracts that they will be interacting with. ABIs are loaded
        using the function :py:func:`verbs.abi.load_abi`.

        Parameters
        ----------
        env: verbs.types.Env
            Simulation environment
        i: int
            Agent index in the simulation
        pool_implementation_abi: type
            abi of the Aave v3 pool contract
        oracle_abi: type
            abi of the Aave oracle contract for collateral and debt tokens
        mintable_erc20_abi: type
            abi of ERC20 contract
        pool_address: bytes
            Addres of Aave v3 pool contract
        oracle_address: bytes
            Address of Aave oracle contract for collateral and debt tokens
        token_a_address: bytes
            Address of collateral token (usually the risky token)
        token_b_address: bytes
            Address of debt token (usually the less risky token)
        activation_rate: float
            Probability of taking an action (either provide collateral
            or borrow) at each step
        """

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

    def update(self, rng: np.random.Generator, env) -> List[verbs.types.Transaction]:
        """
        Update the state of the agent and returns
        list of transactions according to their policy.

        Borrower agent can either supply collateral to the Aave pool
        or borrow debt assets.

        Parameters
        ----------
        rng: numpy.random.Generator
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
        self.step += 1

        if rng.random() < self.activation_rate:
            if not self.has_supplied:
                supply_tx = self.pool_implementation_abi.supply.transaction(
                    self.address,
                    self.pool_address,
                    [self.token_a_address, 10**18, self.address, 0],
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
                u = rng.integers(low=7000, high=9300)
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

    def record(self, env) -> Tuple[int, float, float, float]:
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
        tuple[int, float, float, float]
            Tuple containing:

            - Step of the simulation.
            - Health factor of the borrower's position at the current step.
            - Collateral value of the borrower's position in the base currency
              In Aave the base currency is USD and it has 8 decimal places
            - Debt asset value of the borrower's position in the base currency
        """

        user_data = self.pool_implementation_abi.getUserAccountData.call(
            env, self.address, self.pool_address, [self.address]
        )[0]
        health_factor = user_data[5] / 10**18
        health_factor = min(health_factor, 100)
        collateral_base = user_data[0] / 10**8
        debt_base = user_data[1] / 10**8
        return self.step, health_factor, collateral_base, debt_base
