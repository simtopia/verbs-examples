******************
Uniswap Simulation
******************

Full code for this simulation can be found at https://github.com/simtopia/verbs-examples/blob/main/uniswap_example.py

In this example we model an agent that trades between a Uniswap pool and an external market (modelled by a Geometric
Brownian Motion) in order to make a profit.

.. note::
    We consider the Uniswap v3 pool for WETH and DAI with fee 3000.

    The price of the risky asset (WETH) in terms of the stablecoin (DAI) in the
    external market is modelled by a GBM.

The goal of the simulation is for the price of Uniswap to follow the price
in the external market. This happens when a trader makes arbitrage by trading
in both markets (buying an asset where it is cheaper,
and selling it where it is more expensive).

In each simulation step, the trader makes the the right trade in Uniswap so that
the new price in Uniswap matches the price in the external market. We assume
the external market is frictionless.

Our simulations will obey the following broad structure,


#. Define the EVM simulation environment, see the `VERBS documentation <https://simtopia.github.io/verbs/pages/simulation_environment.html>`_.

#. Prepare the contract addresses that will be used during the simulation.

   .. tip::

      Contracts used during the simulation can be either manually deployed, or can be
      forked from a live deployment.

#. Define the agents' behaviours and initialise them.

#. Run the simulations.


.. toctree::
   :maxdepth: 2

   uniswap_sim
