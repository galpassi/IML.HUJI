import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import polynomial_fitting as pf
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import os
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=[2])
    df = _process_categorials(df)
    df = _process_date(df)
    df = _remove_ilegal(df)

    y = df['Temp']
    X = df.drop('Temp', 1)
    return X, y

def _process_categorials(df):
    #dummies = pd.get_dummies(df['Country'])
    #df = df.drop('Country', 1)
    #for country in dummies.columns:
    #    df.insert(0, country, dummies[country])
    df = df.drop('City', 1)  # one city per country --> doesn't add data
    return df

def _process_date(df):
    df['DayOfYear'] = df['Date'].map(lambda x: x.timetuple().tm_yday)
    df['Date'] = df['Date'].apply(lambda x: datetime.toordinal(x))
    df = df.drop('Day', 1)
    return df

def _remove_ilegal(df):
    return df[df['Temp'] > -12]

def create_is_temp_scatter(data, path=""):
    data = data[data['Country'] == 'Israel']
    data = data[['DayOfYear', 'Year', 'Temp']]
    fig = px.scatter(data, x="DayOfYear", y="Temp", color='Year')
    fig.update_layout(title={
        'text': f"average daily temp in Israel(Tel-Aviv) (y) as a function of the day of the year (x)",
        'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    save_to = os.path.join(path, f"temp_scatter_Israel.html")
    fig.write_html(save_to)

def create_month_bar_plot(data, path=""):
    data = data[data['Country'] == 'Israel']
    data = data[['Month', 'Temp']]
    data = data.groupby('Month', as_index=False).std()
    fig = px.bar(data, x='Month', y='Temp')
    fig.update_layout(title={
        'text': f"STD of temperatures measured in Israel (Tel-Aviv) aggregated over months through 1995-2007 ",
        'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        yaxis_title='Tmp STD')
    save_to = os.path.join(path, f"temp_std_over_months_Israel.html")
    fig.write_html(save_to)

def create_countries_temp_plot(data, path=""):
    data = data[['Country', 'Month', 'Temp']]
    data_mean = data.groupby(['Country', 'Month'], as_index=False).mean()
    data_std = data.groupby(['Country', 'Month'], as_index=False).std()
    data_mean['std'] = data_std['Temp']
    fig = px.line(data_mean, x="Month", y="Temp", color='Country', error_y="std")
    fig.update_layout(title={
        'text': f"Temp mean over months in different countries",
        'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        yaxis_title='Tmp Mean\nSTD in bars')
    save_to = os.path.join(path, f"avg_monthly_temp_in_different_countries.html")
    fig.write_html(save_to)

def _create_k_losses_plot(losses, path=""):
    data = pd.DataFrame({'degree': [k for k, l in losses], 'loss': [l for k, l in losses]})
    fig = px.bar(data, x='degree', y='loss')
    fig.update_layout(title={
        'text': f"MSE losses (y) for polynomial models with different degrees (x)",
        'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        yaxis_title='MSE loss')
    save_to = os.path.join(path, f"polyfit_losses.html")
    fig.write_html(save_to)

def _create_country_losses_plot(losses, path=""):
    data = pd.DataFrame({'country': [c for c, l in losses], 'loss': [l for c, l in losses]})
    fig = px.bar(data, x='country', y='loss')
    fig.update_layout(title={
        'text': f"MSE losses (y) using model trained on israel data (k = 3) to fit different countries (x)",
        'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        yaxis_title='MSE loss')
    save_to = os.path.join(path, f"countries_losses_deg_3.html")
    fig.write_html(save_to)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X, y = load_data(r"..\datasets\City_Temperature.csv")
    full = X.copy()
    full['Temp'] = y
    # Question 2 - Exploring data for specific country
    create_is_temp_scatter(full, "")
    create_month_bar_plot(full, "")

    # Question 3 - Exploring differences between countries
    create_countries_temp_plot(full, "")

    # Question 4 - Fitting model for different values of `k`
    data = full[full['Country'] == 'Israel']
    train_x, train_y, test_x, test_y = split_train_test(data['DayOfYear'].to_frame(), data['Temp'], 0.75)
    losses = []
    for k in range(1,11):
        estimator = pf.PolynomialFitting(k)
        estimator.fit(train_x.to_numpy(), train_y.to_numpy())
        losses.append((k, round(estimator.loss(test_x.to_numpy(), test_y.to_numpy()), 2)))
    print(f"(degree, MSE loss):\n{losses}")
    _create_k_losses_plot(losses, "")

    # Question 5 - Evaluating fitted model on different countries
    estimator = pf.PolynomialFitting(3)
    train_x, train_y, test_x, test_y = split_train_test(data['DayOfYear'].to_frame(), data['Temp'], 1.0)
    estimator.fit(train_x.to_numpy(), train_y.to_numpy())
    losses = []
    for country in ['Jordan', 'South Africa', 'The Netherlands']:
        data = full[full['Country'] == country]
        train_x, train_y, test_x, test_y = split_train_test(data['DayOfYear'].to_frame(), data['Temp'], 0.0)
        losses.append((country, round(estimator.loss(test_x.to_numpy(), test_y.to_numpy()), 2)))
    print(losses)
    _create_country_losses_plot(losses, "")
