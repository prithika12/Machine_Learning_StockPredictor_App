import streamlit as st
from datetime import date
import pandas as pd

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

import requests
from io import StringIO

START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set page configuration to apply custom CSS
st.set_page_config(layout="wide")

# Apply custom CSS to align content to the left
st.markdown("""
    <style>
        .main .block-container {
            max-width: 1500px;
            margin-top: 50px;
            margin-left: 50px;
            margin-right: auto;
            padding: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Function to get the list of stock symbols
@st.cache
def get_sp500_tickers():
   url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
   html = requests.get(url).text
   sp500 = pd.read_html(StringIO(html))
   tickers = sp500[0]['Symbol'].tolist()
   return tickers

# Fetch the list of stock symbols
stocks = get_sp500_tickers()
stocks.insert(0, "Choose an option")

# Function to fetch stock news from Yahoo Finance
@st.cache(suppress_st_warning=True)
def fetch_stock_news(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}?p={symbol}&.tsrc=fin-srch"
    r = requests.get(url)
    news = []
    if r.status_code == 200:
        news.append(r.text)
    return news

st.sidebar.markdown("<h1>Stock News</h1>", unsafe_allow_html=True)

selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)

if selected_stock != "Choose an option":
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_raw_data()

    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Forecast data')
    st.write(forecast.tail())
        
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    # Display stock news in sidebar
    st.sidebar.subheader('Latest Stock News')
    news = fetch_stock_news(selected_stock)
    for article in news:
        st.sidebar.info(article)

else:
    st.write("Please select a stock symbol to see the prediction.")
