import pandas as pd

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from math import atan2, pi
import os
from IMLearn.metrics import accuracy


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(os.path.join(r"..\datasets", filename))
    return data[:, 0:2], data[:, 2]

def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)
        # Fit Perceptron and record loss in each fit iteration
        losses = []
        classifier = Perceptron(callback=lambda p, x_, y_: losses.append(p.loss(X, y)))
        classifier.fit(X, y)
        # Plot figure of loss as function of fitting iteration

        df = pd.DataFrame(dict(losses=losses, iterations=list(range(len(losses)))))
        fig = px.line(df, x="iterations", y="losses")
        fig.update_layout(title={
            'text': f"missclassification loss(y) as a function of precptron iterations - {n}",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
        fig['layout']['title']['font'] = dict(size=20)
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        linear, gaussian = LDA(), GaussianNaiveBayes()
        linear.fit(X, y)
        gaussian.fit(X, y)


        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        df = pd.DataFrame(X, columns=['x', 'y'])
        df['true'] = y
        df['linear_pred'] = linear.predict(X)
        df['gaussian_pred'] = gaussian.predict(X)
        lda_acurracy, gaussian_accuracy = accuracy(y, linear.predict(X)), accuracy(y, gaussian.predict(X))
        print(df)

        # Add traces for data-points setting symbols and colors
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Gaussian classifier acucuracy  - {gaussian_accuracy}",
        f"LDA classifier acucuracy  - {lda_acurracy}"))
        fig.add_trace(
            go.Scatter(x=df["x"], y=df["y"], mode='markers', marker=dict(color=df['gaussian_pred'], symbol=df['true']),
                       showlegend=False), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df["x"], y=df["y"], mode='markers', marker=dict(color=df['linear_pred'], symbol=df['true']),
                       showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text="feature 1", row=1, col=1)
        fig.update_xaxes(title_text="feature 1", row=1, col=2)
        fig.update_yaxes(title_text="feature 2", row=1, col=1)
        fig.update_yaxes(title_text="feature 2", row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=gaussian.mu_[:, 0], y=gaussian.mu_[:, 1], mode='markers', marker=dict(color='black', symbol='x'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=linear.mu_[:, 0], y=linear.mu_[:, 1], mode='markers', marker=dict(color='black', symbol='x'),showlegend=False), row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(gaussian.classes_)):
            fig.add_trace(get_ellipse(gaussian.mu_[i], np.diag(gaussian.vars_[i])), row=1, col=1)
            fig.add_trace(get_ellipse(linear.mu_[i], linear.cov_), row=1, col=2)

        #add dummy traces to customize legend
        fig.add_trace(go.Scatter(y=[None], mode='markers',marker=dict(symbol='circle', color='black'), name='group 0 true',))
        fig.add_trace(go.Scatter(y=[None], mode='markers',marker=dict(symbol='square', color='black'), name='group 1 true',))
        fig.add_trace(go.Scatter(y=[None], mode='markers',marker=dict(symbol='diamond', color='black'), name='group 2 true',))
        fig.add_trace(go.Scatter(y=[None], mode='markers',marker=dict(symbol='circle', color='purple'), name='group 0 - predicted',))
        fig.add_trace(go.Scatter(y=[None], mode='markers',marker=dict(symbol='circle', color='lightblue'), name='group 1 - predicted',))
        fig.add_trace(go.Scatter(y=[None], mode='markers',marker=dict(symbol='circle', color='yellow'), name='group 2 - predicted',))
        fig.add_trace(go.Scatter(y=[None], mode='markers',marker=dict(symbol='x', color='black'), name='predicted group expectation',))
        fig.add_trace(go.Scatter(y=[None], mode='markers',marker=dict(symbol='circle-open', color='black'), name='predicted group covariance',))

        fig.update_layout(title_text=f"Classification of {f[:-4]} color - predicted label, shape - true label", title_x=0.5)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
