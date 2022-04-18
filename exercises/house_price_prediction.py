from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

import os
from typing import NoReturn
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from uszipcode import SearchEngine as zip_engine
pio.templates.default = "simple_white"



def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    df = pd.read_csv(filename)
    df = _remove_negatives(df)
    df = _process_date(df)
    df = _process_yr_built(df)
    df = _process_yr_renovated(df)
    df = _process_zipcode(df)

    #  create responces, padd with 1s for intercept, drop ids, cast to float
    df = df.drop('id', 1)
    df = df.astype('float')
    responces = df['price'].to_frame()
    responces = responces.rename(columns={'price': 'y'})
    df = df.drop('price', 1)
    #df.insert(0, 'ones', 1.0)
    return df, responces

def _remove_negatives(df):
    df = df[df['price'] > 0]
    df = df[df['sqft_living'] > 0]
    df = df[df['sqft_lot'] > 0]
    df = df[df['floors'] > 0]
    df = df[df['sqft_lot15'] > 0]
    return df

def _process_date(df):
    df = df[df['date'].notna()]  # remove nan
    df = df[df['date'] != '0']  # remove 0s
    df['date'] = df['date'].apply(lambda x: datetime.toordinal(datetime(int(str(x)[0:4]), int(str(x)[4:6]), int(str(x)[6:8]))))
    return df

def _process_yr_built(df):
    df = df[df['yr_built'].notna()]   # remove nan
    df = df[df['yr_built'] != '0']  # remove 0s
    df['yr_built'] = df['yr_built'].apply(lambda x: datetime.toordinal(datetime(int(str(x)[0:4]), 1, 1)))
    return df

def _process_yr_renovated(df):
    """
    replace non zero values to correct form and zero values with the year the house was built
    """
    df['yr_renovated'] = df['yr_renovated'].iloc[df['yr_renovated'].to_numpy().nonzero()].map(
        lambda x: datetime.toordinal(datetime(int(x), 1, 1)))
    df['yr_renovated'] = np.where(df['yr_renovated'].isna, df['yr_built'], df['yr_renovated'])
    return df

def _process_zipcode(df):
    """
    change zipcode to median_income and median_house_value from uszipcode package
    """
    engine = zip_engine()
    md_value = df['zipcode'].map(lambda x: float(engine.by_zipcode(int(x)).median_home_value))
    md_income = df['zipcode'].map(lambda x: float(engine.by_zipcode(int(x)).median_household_income))
    df['zipcode'] = md_value
    df = df.rename(columns={'zipcode': 'md_value'})
    df.insert(df.columns.get_loc('md_value'), 'md_income', md_income)
    return df

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_std = y.std(ddof=1)
    for col in X.columns:
        y.insert(0, 'x', X[col])
        x_std = X[col].std(ddof=1)
        pearson = round(float((y.cov(ddof=1)['y'][0]) / (x_std * y_std)), 3)
        fig = px.scatter(y, x="x", y="y")
        fig.update_layout(title={
            'text': f"Correlation between {col} with responses \npearson: {pearson}",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis_title=f"{col}",
            yaxis_title=f"price ($)")
        save_to = os.path.join(output_path, f"{col}.html")
        fig.write_html(save_to)
        y = y.drop('x', 1)

def _write_results_figure(restults, path=""):
    data = pd.DataFrame(dict(percent=list(range(10, 101)), means=[i for i, j in restults],
                             lower=[i - 2 * j for i, j in restults], upper=[i + 2 * j for i, j in restults]))
    fig = go.Figure([
        go.Scatter(
            name='MSE loss',
            x=data['percent'],
            y=data['means'],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=data['percent'],
            y=data['lower'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=data['percent'],
            y=data['upper'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        yaxis_title='MSE loss',
        xaxis_title = r"% of training data",
        title='MSE loss for increasing percentage of training data used',
        hovermode="x"
    )
    fig.write_html(os.path.join(path, 'final_results.html'))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(r"..\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, r"")

    # Question 3 - Split samples into training- and testing sets.
    try:
        #consider dropping data with low pearson corrolation

        X = X.drop('yr_built', 1)
        X = X.drop('yr_renovated', 1)
        X = X.drop('date', 1)
        X = X.drop('sqft_lot15', 1)
        X = X.drop('sqft_lot', 1)
        X = X.drop('sqft_basement', 1)
        #X = X.drop('long', 1)
        #X = X.drop('lat', 1)
        y = y.drop('x', 1)

    except KeyError:
        pass

    train_x, train_y, test_x, test_y = split_train_test(X, y['y'], 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    estimator = LinearRegression()
    train = train_x
    train['y'] = train_y
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()
    results = []
    for p in range(10, 101):
        losses = []
        for i in range(1, 11):
            temp_train = train.sample(frac=round(p/100, 2))
            temp_y = temp_train['y']
            temp_train = temp_train.drop('y', 1)
            estimator.fit(temp_train.to_numpy(), temp_y.to_numpy())
            losses.append(round(estimator.loss(test_x, test_y), 2))

        losses = np.array(losses)
        results.append((losses.mean(), losses.std(ddof=1)))

    _write_results_figure(results, r"")





