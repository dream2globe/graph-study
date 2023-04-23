import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.rank import pagerank, radiorank
from src.utils.graph import build_nx_graph
from src.utils.logger import get_logger

# General
logger = get_logger()
random_seed = 42

# For pagerank
alpha = 0.1  # damping factor
weight = "value"  # weight key in graph
use_pos = False
if use_pos == True:
    pos = pd.read_csv("data/processed/position.csv")  # node positions in graph
else:
    pos = None

# Cache of prediction
## structure : {"y_num_xs" : [("features": set[str], "mse": float)]}
path_cache_mse = Path("data/evaluation/cache_mse.pickle")
if path_cache_mse.exists():
    with open(path_cache_mse, "rb") as f:
        cache_mse = pickle.load(f)
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
    for num_xs in range(start_num, len(trainset.columns), step):
        xs = features[:num_xs]
        ys = features[num_xs:]
        logger.info(f"Input variables: {xs}")
        mses = [(num_xs, y, eval_mse_one(trainset, testset, xs, y, model)) for y in ys]
        mse_records.extend(mses)
    mses_df = pd.DataFrame().from_records(mse_records, columns=["num_xs", "y", "mse"])
    return mses_df


def infer_relation(data: np.array, method: str = "corr") -> np.array | None:
    match (method):
        case "corr":
            return np.corrcoef(data, rowvar=False)
        case "lasso":
            cov = GraphicalLassoCV().fit(data)
            return cov.precision_
        case _:
            logger.error(
                "'method' argument should be one of 'corr(Correlation)', 'lasso(Graphical Lasso)' & 'kl(KL-Divergence)"
            )
            return None
        # case "kl":
        #     pass


if __name__ == "__main__":
    # Load preprocessed data
    logger.info(f"Load preprocessed data")
    with open("data/processed/values.pickle", "rb") as f:
        value_df = pickle.load(f)
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

    corr_matrix = scaled_train.corr()

    # RadioRank
    alphas = np.arange(0.1, 1.1, 0.1)
    logger.info(f"Evaluating prediction performance")
    eval_df_all = pd.DataFrame()
    logger.info(f"Evaluating prediction performance")
    for alpha in tqdm(alphas):
        # Build a graph using a correlation matrix
        G = build_nx_graph(corr_matrix, titles, pos=pos, threshold=0)
        # Features which are ordered by importance
        selected_nodes = radiorank(G, alpha, weight)
        # Evaluating prediction performance
        eval_df_one = eval_mse_all(
            scaled_train,
            scaled_test,
            selected_nodes,
            start_num=1,
            step=2,
            model=lgb.LGBMRegressor(random_state=random_seed),
        )
        eval_df_one["ranker"] = "RadioRank"
        eval_df_one["alpha"] = alpha
        eval_df_all = pd.concat([eval_df_all, eval_df_one])

    # PageRank
    for alpha in tqdm(alphas):
        # Build a graph using a correlation matrix
        G = build_nx_graph(corr_matrix, titles, pos=pos, threshold=0)
        # Features which are ordered by importance
        pr_score = pagerank(G, alpha)
        selected_nodes = pd.Series(pr_score).sort_values(ascending=False).index.tolist()
        # Evaluating prediction performance
        eval_df_one = eval_mse_all(
            scaled_train,
            scaled_test,
            selected_nodes,
            start_num=1,
            step=2,
            model=lgb.LGBMRegressor(random_state=random_seed),
        )
        eval_df_one["ranker"] = "PageRank"
        eval_df_one["alpha"] = alpha
        eval_df_all = pd.concat([eval_df_all, eval_df_one])

    # Save results before the end of S/W running
    with open("data/evaluation/cache_mse.pickle", "wb") as fw:
        pickle.dump(cache_mse, fw)
    with open("data/evaluation/mse.pickle", "wb") as fw:
        pickle.dump(eval_df_all, fw)
