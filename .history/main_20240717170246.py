import streamlit as st # python framework used to create web apps
from datetime import date

import yfinance as yf # python library to fetch financial data from Yahoo Finance
from prophet import Prophet # prophet module for time series forecasting
from prophet.plot import plot_plotly # module to visualize Prophet forecasts using Plotly
from plotly import graph_objs as go # library as go to create interactive graphs

START = "2012-01-01" #data starting from this date
TODAY = date.today().strftime("%Y-%m-%d") #data upto today

st.title("Stock Prediction App")

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365
