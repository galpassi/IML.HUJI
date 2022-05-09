from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
import copy
from ...metrics import loss_functions


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.pi_ = np.array([(y == label).sum() / y.shape[0] for i, label in enumerate(self.classes_)])
        self.mu_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        for i, label in enumerate(self.classes_):
            self.mu_[i] = np.mean(X[np.where(y == label)], axis=0)

        X_ = copy.deepcopy(X)
        for i, label in enumerate(self.classes_):
            X_[np.where(y == label)] = X[np.where(y == label)] - self.mu_[i]

        self.cov_ = np.cov(X_.T, ddof=self.classes_.shape[0])
        #self.cov_ = np.sum([np.outer(X_[i], X_[i]) for i in range(len(X_))], 0) / (len(X_) - len(self.classes_))
        self._cov_inv = np.linalg.inv(self.cov_)

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
        likelihood = self.likelihood(X)
        classes_ind = np.argmax(likelihood, axis=1)
        to_class = np.vectorize(lambda x: self.classes_[x])
        return to_class(classes_ind)


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihoods = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i, label in enumerate(self.classes_):
            l = lambda x: (x.T @ (self._cov_inv @ self.mu_[i])) + \
                          np.log(self.pi_[i]) - 0.5*(self.mu_[i] @ self._cov_inv @ self.mu_[i])
            likelihoods[:, i] = np.apply_along_axis(l, 1, X)

        return likelihoods

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

