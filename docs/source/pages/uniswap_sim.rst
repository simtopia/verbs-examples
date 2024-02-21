**********
Simulation
***********

.. _sec-uniswap:
Simulation environment
=======================

The simulation environment is implemented in Rust with a
Python API to allow Python implemented agents to interact
with the simulated EVM. It also has functionality
to update the state of the simulation, and track logs
generated during execution.

Initialisation of the simulation environment
---------------------------------------------

The EVM state of the simulation environment is stored as
local in memory data structures. This in-memory database
can be initialised in several ways dependent on the use
case.

In this simulation we do the following:

* Create a forked environment

* Initialise and run an exploratory simulation.

* Export the cached requests of the first exploratory simulation.

   .. code-block:: python

        env = verbs.envs.ForkEnv(url, 1234, 1000)
        # Initialise & run a simulation
        ...
        # Export the cached requests
        cache = env.export_cache()
        # Use this cache to initialise a new environment
        faster_env = verbs.envs.EmptyEnv(1234, cache=cache)



Contracts for the simulation
=============================

The simulation requires the contracts stored in the following addresses,
that are converted to ``bytes``.

The simulation code is wrapped in a ``runner()`` function.

.. code-block:: python

    import verbs

    WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
    DAI_ADMIN = "0x9759A6Ac90977b93B58547b4A71c78317f391A28"
    UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
    UNISWAP_WETH_DAI = "0xC2e9F25Be6257c210d7Adf0D4Cd6E3E881ba25f8"
    SWAP_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
    UNISWAP_QUOTER = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"

    def runner(
        env,
        seed: int,
        n_steps: int,
        init_cache: bool = False,
        mu: float = 0.0,
        sigma: float = 0.3
    )

        weth_address = verbs.utils.hex_to_bytes(WETH)
        dai_address = verbs.utils.hex_to_bytes(DAI)
        swap_router_address = verbs.utils.hex_to_bytes(SWAP_ROUTER)
        quoter_address = verbs.utils.hex_to_bytes(UNISWAP_QUOTER)
        dai_admin_address = verbs.utils.hex_to_bytes(DAI_ADMIN)

        ...


The following example shows how to use the VERBS functionality to
encode / decode data and interact with the EVM via contract functions.
In this snippet the call the ``getPool`` function of
the **UniswapV3 Factory** contract
to get the address of the WETH-DAI pool with fee 3000.

.. code-block:: python

    def runner(...):

        ...

        fee = 3000
        pool_address = abi.uniswap_factory.getPool.call(
            env,
            verbs.utils.ZERO_ADDRESS,
            verbs.utils.hex_to_bytes(UNISWAP_V3_FACTORY),
            [WETH, DAI, fee],
        )[0][0]

        assert pool_address == UNISWAP_WETH_DAI.lower()
        pool_address = verbs.utils.hex_to_bytes(pool_address)


Uniswap trader
===============

The next step is to define the behaviour of the Uniswap trader that
trades between Uniswap and an external market.

.. note::

    The external market provides the price of WETH-DAI and
    for simplicity is modelled as a Geometric Brownian Motion.


In each step, the trader observes the price in Uniswap, ``sqrt_price_uniswap_x96``,
the liquidity in the current tick range, ``liquidity``,
and the price in the external market ``sqrt_target_price_x96``.

The Uniswap agent follows the following logic to find the right trade such that
after it the new ``sqrt_price_uniswap_x96`` is the same as ``sqrt_target_price_x96``.


#. First, it gets an approximated trade, taking into account that in
   Uniswap v2 (or v3 if there is not a tick range change),
   we have :math:`L = \frac{\Delta token1}{\Delta \sqrt{P}}`
   where :math:`token1` is the numeraire and P is the price of :math:`token0`
   in terms of `token1`.


#. The above calculation does not take into account possible tick range
   changes after the trade, with the subsequent change in liquidity. Hence
   the agent makes an optimization using the ``root_scalar`` function
   in order to find the right trade.

.. warning::
    The Uniswap fees / gas fees paid for the trade resulting from the above calculation
    might swipe the possible arbitrage opportunities.
    Nevertheless the above is still useful to simulate a GBM in a Uniswap pool.

The following code provides the above funcitonality when ``sqrt_price_uniswap_x96 < sqrt_target_price_x96``.
Full implementation of the agent is `here <https://github.com/simtopia/verbs-examples/blob/main/simulations/agents/uniswap_agent.py>`_,
including the external market as a Geometric Brownian Motion.

.. code-block:: python

    class UniswapAgent:

        ...
        def get_swap_size_to_increase_uniswap_price(
            self,
            env,
            sqrt_target_price_x96: int,
            sqrt_price_uniswap_x96: int,
            liquidity: int,
            exact: bool = True,
        ):
            """
            Gets the swap parameters so that, after the swap, the price in Uniswap
            is the same as the target price.
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

Full implementation of the Uniswap trader is `here <https://github.com/simtopia/verbs-examples/blob/main/simulations/agents/uniswap_agent.py>`_.

Next, we initialise the Uniswap trader and we mint enough WETH and DAI
for the trader to use during the simulation.
The trader will
only send transactions through the `Swap Router <https://etherscan.io/address/0xE592427A0AEce92De3Edee1F18E0157C05861564>`_
contract, hence the agent needs to approve this contract to use their tokens:

.. code-block:: python

    def runner(...):

        ...
        uniswap_agent = UniswapAgent(...)

        # mint and approve tokens for the Uniswap agent
        # - Mint DAI and WETH
        # - Approve the Swap Router to use these in their transactions
        mint_and_approve_weth(
            env=env,
            weth_abi=abi.weth_erc20,
            weth_address=weth_address,
            recipient=uniswap_agent.address,
            contract_approved_address=swap_router_address,
            amount=int(1e24),
        )
        mint_and_approve_dai(
            env=env,
            dai_abi=abi.dai,
            dai_address=dai_address,
            contract_approved_address=swap_router_address,
            dai_admin_address=dai_admin_address,
            recipient=uniswap_agent.address,
            amount=int(1e30),
        )

where we use the functions ``mint_and_approve_weth`` and ``mint_and_approve_dai``
defined  `here <https://github.com/simtopia/verbs-examples/blob/main/simulations/utils/erc20.py>`_.

Running the simulation
=======================

The environment and agents are wrapped in a :py:class:`verbs.sim.Sim`
and then we can run the simulation. If we are running the initial simulation,
the cache is saved.

.. code-block:: python

    def runner(...)
        ...
        runner = verbs.sim.Sim(seeds, env, [uniswap_agent])
        results = runner.run(n_steps=n_steps)
        if init_cache:
            cache = env.export_cache()
            with open(f"{PATH_CACHE}/cache.json", "w") as f:
                json.dump(verbs.utils.cache_to_json(cache), f)
        return results



The sim runner returns a list of records for each agent at every step
of the simulation.


Batch execution from cache
---------------------------
Typically we might want to execute batches of simulation across
random seeds and simulation parameter samples,
:py:meth:`verbs.sim.batch_runner.batch_run`
implements functionality to generate simulation samples in parallel.

The simulation environments for the samples can be initialised from
a cache (generated using the :py:meth:`verbs.envs.ForkEnv.export_cache` method
as seen in the above code snippet).

Batch execution requires a simulation execution function with the signature

.. code-block:: python

   def runner(
       env, seed, n_steps, **params, **sim_kwargs
   ) -> typing.Any:
       ...

We use the Uniswap simulation :py:meth:`runner` function that we have created
to run ``n_samples`` simulations across different values for the GBM drift
and volatility, :math:`\mu, \sigma` as follows

.. code-block:: python

    parameters_samples = [
        dict(mu=mu, sigma=sigma)
        for mu, sigma in product([0.0, 0.1, -0.1], [0.1, 0.2, 0.3])
    ]

    with open(f"{PATH_CACHE}/cache.json"), "r") as f:
        cache_json = json.load(f)
    cache = verbs.utils.cache_from_json(cache_json)

    batch_results = verbs.batch_runner.batch_run(
        runner,
        n_steps=100,
        n_samples=10,
        parameters_samples=parameters_samples,
        cache=cache,
    )

The batch-runner will generate sample and random seed combinations, and
execute simulation across these combinations in parallel. In this example
it will generate 10 Monte-Carlo samples for each set of parameters (90
samples, 9 parameter sets x 10 random seeds) each run for 100 steps.

For convenience the results are returned grouped by the parameters used to
generate them, in this case they will have the structure

.. code-block:: python

   [
       {
           "params": {"mu": 0.0, "sigma":0.1},
           "samples": [
               # List of Monte-Carlo sample results
               ...
           ]
       },
       {
           "params": {"mu": 0.0, "sigma":0.2},
           "samples": [
               # List of Monte-Carlo sample results
               ...
           ]
       }
   ]
