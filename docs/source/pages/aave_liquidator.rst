Aave Liquidator
================

The structure of the simulation is very similar to the :doc:`/pages/uniswap`,
so we will focus on the liquidator logic, namely the liquidator agent checks
whether a liquidation is profitable before making the liquidation call:

* They check the amount of collateral that they would get by liquidation a
  fraction of a loan.

* They check the price of the trade in Uniswap necessary to close the short
  position in the debt asset.

* If they get a profit after closing their short position in the debt asset,
  then they make the transaction.

These three steps are coded in the ``accountability()`` method of the :py:class:`LiquidatorAgent`

.. code-block:: python

   def accountability(self, env, liquidation_address, amount: int) -> bool:
       """
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

Let's break down the code in terms of the actions that the Liquidator takes:

* Check the amount of collateral that they would get by liquidation a
  fraction of a loan.

.. code-block:: python

    def accountability(self, env, liquidation_address, amount: int) -> bool:
        """
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


        ...

In the above snippet the agent tries to get the liquidation event by *calling* (and not executing)
the ``liquidationCall()`` function of the `Aave pool contract <https://etherscan.io/address/0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2#code>`_.
The :py:meth:`verbs.abi.Function.call` function of the ``abi`` object returns the tuple ``(results, logs, gas)``.
In this case we are interested in inspecting the event ``LiquidationCall``, which
is the last event returned by the ``liquidationCall()`` function.

.. tip::
    Events types returned by calling a function can be checked by looking at the solidity code of the contract.

The abi of the Aave pool contract defines the ``LiquidationCall`` event as follows, so we extract the
first and second non-indexed elements of the event, ``debttocover, liquidatedCollateralAmount``,
which are the values the liquidator is interested in.

.. code-block:: json

  {
    "anonymous": false,
    "inputs": [
      {
        "indexed": true,
        "internaltype": "address",
        "name": "collateralasset",
        "type": "address"
      },
      {
        "indexed": true,
        "internaltype": "address",
        "name": "debtasset",
        "type": "address"
      },
      {
        "indexed": true,
        "internaltype": "address",
        "name": "user",
        "type": "address"
      },
      {
        "indexed": false,
        "internaltype": "uint256",
        "name": "debttocover",
        "type": "uint256"
      },
      {
        "indexed": false,
        "internaltype": "uint256",
        "name": "liquidatedcollateralamount",
        "type": "uint256"
      },
      {
        "indexed": false,
        "internaltype": "address",
        "name": "liquidator",
        "type": "address"
      },
      {
        "indexed": false,
        "internaltype": "bool",
        "name": "receiveatoken",
        "type": "bool"
      }
    ],
    "name": "liquidationcall",
    "type": "event"
  },


The next actions that the Liquidator takes are

* Check the price of the trade in Uniswap necessary to close the short
  position in the debt asset.

* If they get a profit after closing their short position in the debt asset,
  then they make the transaction.

.. code-block:: python

   def accountability(self, env, liquidation_address, amount: int) -> bool:

       ...

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

The liquidator is calling the ``quoteExactOutputSingle()`` function of the `Uniswap quoter v2 <https://github.com/Uniswap/v3-periphery/blob/main/contracts/lens/QuoterV2.sol>`_.
The liquidator retrieves the first value of the output of ``quoteExactOutputSingle()``, as it is
the amount of collateral tokens that they would have to pay in order
to recover the the debt assets they spent in the liquidation. The abi of ``quoteExactOutputSingle()`` indicates
the values returned by this function:

.. code-block:: json

  "name": "quoteExactOutputSingle",
  "outputs": [
    {
      "internalType": "uint256",
      "name": "amountIn",
      "type": "uint256"
    },
    {
      "internalType": "uint160",
      "name": "sqrtPriceX96After",
      "type": "uint160"
    },
    {
      "internalType": "uint32",
      "name": "initializedTicksCrossed",
      "type": "uint32"
    },
    {
      "internalType": "uint256",
      "name": "gasEstimate",
      "type": "uint256"
    }
  ],
  "stateMutability": "nonpayable",
  "type": "function"

The liquidation is profitable if the amount of collateral tokens received from the liquidation,
(``liquidated_collateral_amount``), is greater than the amount of collateral token spent in the
swap (``amount_collateral_from_swap``) to recover the amount debt tokens spent in the liquidation.

Full implementation of the Liquidator agent is `here <https://github.com/simtopia/verbs-examples/blob/main/verbs_examples/agents/liquidation_agent.py>`__.
