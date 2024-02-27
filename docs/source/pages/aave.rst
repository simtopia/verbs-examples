***************
AAVE Simulation
***************

In this example we consider the interaction between Aave and Uniswap:

* A `Uniswap Agent` that trades between a Uniswap pool and
  an external market, modelled by a Geometric Brownian Motion, in order to make a profit.

  .. note::
      We consider the Uniswap v3 pool for WETH and DAI with fee 3000.

      The price of the risky asset (WETH) in terms of the stablecoin (DAI) in the
      external market is modelled by a GBM.

* Several `Borrow Agents` that borrow DAI from an Aave v3 pool and deposit WETH as collateral.

* A `Liquidation Agent` that liquidates those positions from the `Borrow agents` that are
  in distress (that is, their Health Factors are < 1) as long as the liquidation is
  profitable for the liquidation agent.

  .. tip::

     The liquidator agent checks whether a liquidation is profitable before making
     the liquidation call:

     * They check the amount of collateral that they would get by liquidation a
       fraction of a loan.

     * They check the price of the trade in Uniswap necessary to close the short
       position in the debt asset.

     * If they get a profit after closing their short position in the debt asset,
       then they make the transaction.

Full code for this simulation is in https://github.com/simtopia/verbs-examples/blob/main/aave_example.py

Reference for this simulation: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4540333

.. toctree::
   :maxdepth: 2

   aave_liquidator
