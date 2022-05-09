from __future__ import annotations
from typing import Callable
from typing import NoReturn
from ...base import BaseEstimator
from ...metrics import loss_functions
import numpy as np


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
    """
    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """
        def __batchPerceptron(samples, labels, iterations):
            w = np.zeros(samples.shape[1])
            self.coefs_, self.fitted_ = w, True
            while iterations > 0:
                y_pred = samples @ w
                y_pred[np.where(y_pred < 0)] = -1
                y_pred[np.where(y_pred >= 0)] = 1
                loss = loss_functions.misclassification_error(labels, y_pred, False)
                if loss == 0:
                    self.coefs_ = w
                    return
                wrong_index = np.where(labels * y_pred < 1)[0][0]
                w = w + (labels[wrong_index]*samples[wrong_index])
                self.coefs_ = w
                self.callback_(self, samples[wrong_index], labels[wrong_index])
                iterations -= 1

            self.coefs_ = w

        if self.include_intercept_:
            X = np.insert(X, 0, [1] * X.shape[0], axis=1)
        __batchPerceptron(X, y, self.max_iter_)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            X = np.insert(X, 0, [1] * X.shape[0], axis=1)
        pred = X @ self.coefs_
        pred[np.where(pred < 0)] = -1
        pred[np.where(pred >= 0)] = 1
        return pred

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return loss_functions.misclassification_error(y, self._predict(X))

