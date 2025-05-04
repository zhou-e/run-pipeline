from multiprocessing import cpu_count, Pool

import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.constants import FEATURE_KEY, TARGET_KEY

# Global variable for multiprocessing
_global_target = None


def init_worker(target):
    global _global_target
    _global_target = target


def _col_iter(df, cols):
    for col in cols:
        yield col, df[col].values.reshape(-1, 1)


def screen_param(args, threshold=0.1):
    name, values = args
    reg = LinearRegression().fit(values, _global_target)
    if reg.score(values, _global_target) > threshold:
        return name


def screen(df: pd.DataFrame, df_json: dict, n_cpu: int = None):
    if n_cpu is None:
        n_cpu = cpu_count() - 1

    features = df_json[FEATURE_KEY]
    target_col = df_json[TARGET_KEY]
    target = df[target_col].values.reshape(-1, 1)

    with Pool(n_cpu, initializer=init_worker, initargs=(target,)) as p:
        res = p.map(screen_param, _col_iter(df, features))

    features = [r for r in res if r]
    df_json[FEATURE_KEY] = features

    return df, df_json
