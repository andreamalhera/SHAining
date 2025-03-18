import pandas as pd
import re
from itertools import product
from shaining.utils.param_keys.generator import LOG

def prune_per_benchmarking(df: pd.DataFrame, miners: list, metrics: list=[]) -> pd.DataFrame:
    if len(metrics) == 0:
        cross_product = [column for miner in miners for column in df.columns if column.endswith(miner)]
    else:
        cross_product = [f"{a}_{b}" for a, b in product(metrics, miners)]
    pruned_df = df[[LOG, *cross_product]].dropna()
    return pruned_df

def prune_ELnames_per_level(names: list, level: int)-> list:
    pruned_names = [
        item for item in names
        if sum(1 for part in re.sub(r'^genEL\d+_', '', item).split('.')[0].split('_') if part != "nan")==level
        ]
    return pruned_names
