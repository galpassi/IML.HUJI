from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
import pandas as pd
import statistics

from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score, validation_score = 0.0, 0.0
    X_ = pd.DataFrame(X)
    X_['y_'] = y
    X_ = X_.sample(frac=1)
    partitions = np.array_split(X_, cv)
    for k in range(cv):
        train, test = pd.concat(partitions[:k] + partitions[(k+1):]), partitions[k]
        train_y = train["y_"].to_numpy()
        train_x = train.drop("y_", axis=1).to_numpy()
        test_y = test["y_"].to_numpy()
        test_x = test.drop("y_", axis=1).to_numpy()
        estimator.fit(train_x, train_y)
        train_score += scoring(train_y, estimator.predict(train_x))
        validation_score += scoring(test_y, estimator.predict(test_x))
    return train_score / cv, validation_score / cv



