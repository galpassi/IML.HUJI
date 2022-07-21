from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error as mse
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, RidgeRegression
from IMLearn.learners.regressors import LinearRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    poly = lambda x: (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    X = np.random.uniform(-1.2, 2, n_samples)
    y_noiseless = poly(X)
    y = y_noiseless + np.random.normal(0, noise, n_samples)
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion=2/3)

    scatter_data = pd.DataFrame()
    scatter_data['x'] = X
    scatter_data['y1'] = y_noiseless
    scatter_data['y2'] = y
    fig = make_subplots(rows=1, cols=1)
    fig.add_traces([go.Scatter(x=X, y=y_noiseless,   mode='markers', marker=dict(color="red"),  name="noiseless data"),
                 go.Scatter(x=X, y=y, mode='markers', marker=dict(color="blue"), name="noisy data")],
    rows=1, cols=[1, 1])
    fig.update_layout(title={
        'text': f"Scatter plot of data (n={n_samples}) sampled uniformly between [-1.2, 2.0] and labaled using polynomial:\n"
                f"(x+3)*(x+2)*(x+1)*(x-1)*(x-2). Noise level = {noise}",
        'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        yaxis_title='labels')
    fig.write_html(f"q1_scatter_nsamples_{n_samples}_noise_{noise}.html")

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors, validation_errors = [], []
    for k in range(11):
        estimator = PolynomialFitting(k)
        train_err, validation_err = cross_validate(estimator, train_x.to_numpy(), train_y.to_numpy(), mse, 5)
        train_errors.append(train_err)
        validation_errors.append(validation_err)

    #print(train_errors)
    #print(validation_errors)
    data = pd.DataFrame()
    data['degree'] = list(range(11))
    data['train'] = train_errors
    data['validation'] = validation_errors
    fig = px.line(data, x="degree", y=['train', 'validation'])
    fig.update_layout(title={
        'text': f"polynomial fitting 5-folds cross-validation over different polynomial degrees (x).\n"
                f" n={n_samples}, noise={noise}",
        'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        yaxis_title='mse error')
    fig.write_html(f"q1_CV_nsamples_{n_samples}_noise_{noise}.html")

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = validation_errors.index(min(validation_errors))
    print(f"k={k}")
    estimator = PolynomialFitting(k=k)
    estimator.fit(train_x.to_numpy(), train_y.to_numpy())
    print(f"validation error for k={k} --> {round(min(validation_errors), 2)}")
    print(f"test error k={k}: {round(mse(test_y.to_numpy(), estimator.predict(test_x.to_numpy())),2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    data = datasets.load_diabetes()
    data = pd.DataFrame(data=np.c_[data['data'], data['target']],
                         columns=data['feature_names'] + ['resp'])
    train, test = data.iloc[:n_samples, :], data.iloc[n_samples:, :]
    train_y, train_x, test_y, test_x = train['resp'].to_numpy(), train.drop('resp', 1).to_numpy()\
        , test['resp'].to_numpy(), test.drop('resp', 1).to_numpy()


    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    train_err_ridge, validation_err_ridge, train_err_lasso, validation_err_lasso = [], [], [], []
    for i in np.linspace(0, 3, n_evaluations):
        estimator_lasso, estimator_ridge = Lasso(alpha=i), RidgeRegression(lam=i),
        l1_train, l1_validation = cross_validate(estimator_lasso, train_x, train_y, mse, cv=5)
        l2_train, l2_validation = cross_validate(estimator_ridge, train_x, train_y, mse, cv=5)
        train_err_lasso.append(l1_train)
        validation_err_lasso.append(l1_validation)
        train_err_ridge.append(l2_train)
        validation_err_ridge.append(l2_validation)

    errors_plot = pd.DataFrame()
    errors_plot['lambda'] = np.linspace(0, 3, n_evaluations)
    errors_plot['train_l1'] = train_err_lasso
    errors_plot['validation_l1'] = validation_err_lasso
    errors_plot['train_l2'] = train_err_ridge
    errors_plot['validation_l2'] = validation_err_ridge
    fig = px.line(errors_plot, x="lambda", y=['train_l1', 'validation_l1', 'train_l2', 'validation_l2'])
    fig.update_layout(title={
        'text': f"Cross-validation train and test errors for lambda values in range [0-3] of l1 and l2 regularization",
        'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        yaxis_title='mse')
    fig.write_html(f"q2_train_and_validation_errors(0-100).html")


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    min_validation_l1, min_validation_l2 = min(validation_err_lasso), min(validation_err_ridge)
    l1_lambda, l2_lambda = list(np.linspace(0, 3, n_evaluations))[validation_err_lasso.index(min_validation_l1)], \
                           list(np.linspace(0, 3, n_evaluations))[validation_err_ridge.index(min_validation_l2)]
    print(f"l1_lambda={l1_lambda} with validation loss {min_validation_l1}")
    print(f"l2_lambda={l2_lambda} with validation loss {min_validation_l2}")

    estimator_lasso, estimator_ridge, estimator_regression = Lasso(alpha=l1_lambda).fit(train_x, train_y), \
                                                             RidgeRegression(lam=l2_lambda).fit(train_x, train_y), \
                                                             LinearRegression().fit(train_x, train_y)
    print(f"test lasso lambda={l1_lambda}: {round(mse(test_y, estimator_lasso.predict(test_x)), 2)}")
    print(f"test ridge lambda={l2_lambda}: {round(mse(test_y, estimator_ridge.predict(test_x)), 2)}")
    print(f"test regression: {round(mse(test_y, estimator_regression.predict(test_x)), 2)}")



if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0.)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter(n_evaluations=300)
