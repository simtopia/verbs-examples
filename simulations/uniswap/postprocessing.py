import os
from functools import reduce
from typing import List, Tuple

import pandas as pd


def save(batch_results: List[List[Tuple[int, int]]]):

    results_dir = "results/sim_uniswap_gbm"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    df_list = []
    for batch in batch_results:
        params = batch["params"]
        for seed, sample in enumerate(batch["samples"]):
            # flatten the records of all the agents
            flattened_sample = [reduce(lambda x, y: x + y, record) for record in sample]
            df_sample = pd.DataFrame(flattened_sample)
            df_sample = df_sample.assign(seed=seed)
        df_sample = df_sample.assign(**params)
        df_list.append(df_sample)
    df = pd.concat(df_list)
    df.to_parquet(os.path.join(results_dir, "batch_results.parquet"))
