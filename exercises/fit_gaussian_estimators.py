from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
pio.templates.default = "simple_white"

TEST_MU = np.array([0, 0, 4, 0])
TEST_COV = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])


def test_univariate_gaussian():
    estimator = UnivariateGaussian()
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    estimator.fit(samples)
    print(f"({round(estimator.mu_, 3)}, {estimator.var_:.3f})")
    # Question 2 - Empirically showing sample mean is consistent
    distances = [abs(10 - estimator.fit(samples[0:i]).mu_) for i in range(10, 1000, 10)]
    df = pd.DataFrame(dict(n_samples=list(range(10, 1000, 10)), distance=distances))
    fig = px.line(df, x="n_samples", y="distance")
    fig.update_layout(title={
        'text': f"distance (in abs) between estimated means of increased sample sizes to real distribution's expectation",
        'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    fig['layout']['title']['font'] = dict(size=20)

    fig.show()
    # Question 3 - Plotting Empirical PDF of fitted model
    sorted_sample = np.sort(samples)
    pdfs = estimator.pdf(sorted_sample)
    df = pd.DataFrame(dict(sorted_samples=sorted_sample, Gaussian_PDF=pdfs))
    fig = px.scatter(df, x="sorted_samples", y="Gaussian_PDF")
    fig.update_layout(title = {'text': f"gaussian PDF function with mu={estimator.mu_:.2f}, var={estimator.var_:.2f} for each samples value (x)",
                               'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    fig['layout']['title']['font'] = dict(size=20)
    fig.show()


def test_multivariate_gaussian():
    estimator = MultivariateGaussian()
    # Question 4 - Draw samples and print fitted model
    samples = np.random.multivariate_normal(TEST_MU, TEST_COV, 1000)
    estimator.fit(samples)
    print(f"{estimator.mu_[0]:.3f}, {estimator.mu_[1]:.3f}, {estimator.mu_[2]:.3f}, {estimator.mu_[3]:.3f}")
    print(np.around(estimator.cov_, decimals=3))

    # Question 5 - Likelihood evaluation
    values = np.linspace(-10, 10, 200)
    f_13, res = np.array(np.meshgrid(values, values)).T.reshape(-1, 2), []
    for f_1, f_3 in f_13:
        mu = np.array([f_1, 0, f_3, 0])
        res.append(estimator.log_likelihood(mu, TEST_COV, samples))

    res = np.array(res).reshape(200, 200)
    fig = px.imshow(res, labels=dict(x="f3", y="f1"), x=values, y=values)
    fig.update_layout(title="log likelihood calculated over 1000 sampled points witn mean = [0, 0, 4, 0] and fixed covarience matrix \n using mean of [f1, 0, f3, 0] wehre f1,f2 in [-10, 10]")
    fig['layout']['title']['font'] = dict(size=20)
    fig.show()

    # Question 6 - Maximum likelihood
    x, y = np.unravel_index(res.argmax(), res.shape)
    print(f"f1 = {round(values[x], 4)}, f3 = {round(values[y], 4)}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

