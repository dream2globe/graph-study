import pickle
import warnings
from collections import defaultdict
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.rank import pagerank, radiorank
from src.utils.graph import build_nx_graph
from src.utils.logger import get_logger

# General
logger = get_logger()
weight = "value"
random_seed = 42
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# For pagerank
alpha = 0.1  # damping factor
use_pos = False
if use_pos == True:
    pos = pd.read_csv("data/processed/position.csv")  # node positions in graph
else:
    pos = None

# Cache of prediction
## structure : {"y_num_xs" : [("features": set[str], "mse": float)]}
path_cache_mse = Path("data/evaluation/cache_mse.pickle")
if path_cache_mse.exists():
    cache_mse = pd.read_pickle(path_cache_mse)
else:
    cache_mse = defaultdict(list)


def eval_mse_one(
    trainset: pd.DataFrame,
    testset: pd.DataFrame,
    xs: list[str],
    y: str,
    model: any,
) -> float:
    global cache_mse
    key_name = model.__class__.__name__ + "_" + str(y) + "_" + str(len(xs))
    # Search for MSE in cache using the feature names
    recodes = cache_mse.get(key_name)
    if recodes is not None:
        for features, mse in recodes:
            if features == set(xs):
                return mse
    # If MSE doesn't exist in the cache, the prediction logic will run
    x_train, y_train = trainset[xs].values, trainset[y].values
    x_test, y_test = testset[xs].values, testset[y].values
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    cache_mse[key_name].append((set(xs), mse))  # save cache
    return mse


def eval_mse_all(
    trainset: pd.DataFrame,
    testset: pd.DataFrame,
    features: list[str],
    start_num: int,
    step: int,
    model: any,
) -> pd.DataFrame:
    mse_records = []
    p_bar = tqdm(range(start_num, len(trainset.columns), step), leave=False)
    for num_xs in p_bar:
        p_bar.set_description(f"The num of features: {num_xs}")
        xs = features[:num_xs]
        ys = features[num_xs:]
        # logger.info(f"Input variables: {xs}")
        mses = [
            (num_xs, y, eval_mse_one(trainset, testset, xs, y, model))
            for y in tqdm(ys, leave=False)
        ]
        mse_records.extend(mses)
    mses_df = pd.DataFrame().from_records(mse_records, columns=["num_xs", "y", "mse"])
    return mses_df


if __name__ == "__main__":
    # Load preprocessed data
    logger.info(f"Load preprocessed data")
    value_df = pd.read_pickle("data/processed/values.pickle")
    test_items = value_df.columns.tolist()
    item_2_idx = {v: k for k, v in enumerate(test_items)}
    idx_2_item = {k: v for k, v in enumerate(test_items)}

    # Prepare training and test data
    logger.info(f"Prepare training and test data")
    train, test = train_test_split(value_df, test_size=0.5, random_state=random_seed, shuffle=False)
    scaler = StandardScaler().fit(train)
    scaled_train = pd.DataFrame(scaler.transform(train))
    scaled_test = pd.DataFrame(scaler.transform(test))
    titles = train.columns.tolist()

    # Prepare matrix to build a graph
    logger.info(f"build graph using one of 'correlation', 'graph lasso' and 'KL-Divergence'")
    corr_mat = scaled_train.corr()
    cov = GraphicalLassoCV().fit(scaled_train)
    lasso_mat = pd.DataFrame(cov.precision_)

    # Set parameters
    hyper_params = {
        "relation": [
            {"name": "corr_mat", "value": corr_mat},
            {"name": "lasso_mat", "value": lasso_mat},
        ],
        "rankers": [
            {"name": "pagerank", "value": pagerank},
            {"name": "radiorank", "value": radiorank},
        ],
        "alphas": np.arange(0.1, 1.1, 0.1),
        "models": [
            {
                "name": "rf",
                "value": RandomForestRegressor(random_state=random_seed, n_jobs=-1),
            },
            {
                "name": "lgb",
                "value": lgb.LGBMRegressor(random_state=random_seed, n_jobs=-1),
            },
        ],
    }
    combination = product(*[value for value in hyper_params.values()])

    # Run main loop
    eval_df_all = pd.DataFrame()
    p_bar = tqdm(list(combination))
    for relation, ranker, alpha, model in p_bar:
        p_bar.set_description(f"{relation['name']}, {ranker['name']}, {alpha}, {model['name']}")

        # Build a graph using a correlation matrix
        G = build_nx_graph(relation["value"], titles, pos=pos, threshold=0)

        # Features which are ordered by importance
        selected_nodes = ranker["value"](G, alpha, max_iter=500)

        # Evaluating prediction performance
        eval_df_one = eval_mse_all(
            scaled_train,
            scaled_test,
            selected_nodes,
            start_num=1,
            step=2,
            model=model["value"],
        )
        eval_df_one["relation"] = relation["name"]
        eval_df_one["ranker"] = ranker["name"]
        eval_df_one["alpha"] = alpha
        eval_df_one["model"] = model["name"]
        eval_df_all = pd.concat([eval_df_all, eval_df_one])

        # Save prediction performance (overwrite on every iteration)
        with open("data/evaluation/cache_mse.pickle", "wb") as fw:
            pickle.dump(cache_mse, fw)
        with open("data/evaluation/mse.pickle", "wb") as fw:
            pickle.dump(eval_df_all, fw)
