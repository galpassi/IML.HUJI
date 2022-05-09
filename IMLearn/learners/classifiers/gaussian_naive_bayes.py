from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import inv, det, slogdet
from ...metrics import loss_functions



class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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

        self.vars_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        for i, label in enumerate(self.classes_):
            vars = np.array([np.var(X[np.where(y == label)][:, i], ddof=1) for i in range(X.shape[1])])
            self.vars_[i] = vars

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
            sign, logdet = slogdet(np.diag(self.vars_[i]))
            cov_det = sign * np.exp(logdet)
            d = X[:, np.newaxis, :] - self.mu_[i]
            mahalanobis = np.sum(d.dot(inv(np.diag(self.vars_[i]))) * d, axis=2).flatten()
            likelihoods[:, i] = np.exp(-.5 * mahalanobis) / np.sqrt((2*np.pi) ** X.shape[1] * cov_det) * self.pi_[i]

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
