import streamlit as st # python framework used to create web apps
from datetime import date
import pandas as pd #library for data analysis and manipulation

import yfinance as yf # python library to fetch financial data from Yahoo Finance
from prophet import Prophet # prophet module for time series forecasting
from prophet.plot import plot_plotly # module to visualize Prophet forecasts using Plotly
from plotly import graph_objs as go # library as go to create interactive graphs

import requests #library to handle http requests 
from io import StringIO # Library to handle string input/output

START = "2012-01-01" #data starting from this date
TODAY = date.today().strftime("%Y-%m-%d") #data upto today

# Function to get the list of stock symbols
@st.cache_data
def get_sp500_tickers():
   url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
   html = requests.get(url).text # Send an HTTP GET request to the URL and retrieve the HTML content as a string
   sp500 = pd.read_html(StringIO(html)) # Use pandas to read the HTML content and convert it into a list of DataFrames
   tickers = sp500[0]['Symbol'].tolist() # Extract the 'Symbol' column from the first DataFrame and convert it to a list
   return tickers

# Fetch the list of stock symbols
stocks = get_sp500_tickers()
stocks.insert(0, "Choose an option") # Add a placeholder option

st.title("Stock Prediction App")

# Dropdown selection box for choosing a stock symbol
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Check if a valid stock symbol is selected
if selected_stock != "Choose an option":
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)