import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from util.logger import get_logger

logger = get_logger()


def predict_one(
    trainset: pd.DataFrame,
    testset: pd.DataFrame,
    xs: list[str],
    y: str,
    random_seed: int = 42,
) -> float:
    x_train, y_train = trainset[xs].values, trainset[y].values
    x_test, y_test = testset[xs].values, testset[y].values
    model = lgb.LGBMRegressor(random_state=random_seed)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return mean_squared_error(y_test, y_pred)


def predict_all(
    feature_rank: list[str],
    trainset: pd.DataFrame,
    testset: pd.DataFrame,
    min_features: int,
    step: int,
) -> list[tuple]:
    eval_mat = []
    for num_ft in range(min_features, len(trainset.columns), step):
        logger.info(f"current loop: {num_ft}")
        xs = feature_rank[:num_ft]
        ys = feature_rank[num_ft:]
        scores = [predict_one(trainset, testset, xs, y) for y in ys]
        eval_mat.append((num_ft, np.mean(scores), max(scores), min(scores)))
    return eval_mat
