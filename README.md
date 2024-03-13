# VERBS Examples

Example models implemented using [VERBS](https://github.com/simtopia/verbs)

Full documentation for these examples can be found
[here](https://simtopia.github.io/verbs-examples/)

## Installation & Running

This repo uses [hatch](https://hatch.pypa.io/latest/) for dependency
management. The examples can the be run using

```
hatch run examples:uniswap
```

or

```
hatch run examples:aave
```

You can also use the `--help` argument to see additional arguments
to the scripts.

The package can also be imported to run the simulations, e.g.

```
from verbs_examples.aave import sim

...
results = run_from_cache(seed, n_steps, n_borrow_agents, sigma)
```
