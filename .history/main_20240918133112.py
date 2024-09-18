import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import requests
from io import StringIO
import feedparser
import os

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
@st.cache_data
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    sp500 = pd.read_html(StringIO(html))
    tickers = sp500[0]['Symbol'].tolist()
    return tickers

# Function to fetch and parse RSS feed from Yahoo Finance
def fetch_yahoo_finance_news():
    feed_url = "https://finance.yahoo.com/rss/"
    feed = feedparser.parse(feed_url)
    return feed.entries

# Function to get images from a local folder
def get_local_images():
    images = []
    images_folder = 'images'  # Folder containing your images
    for i in range(1, 11):  # Assuming you have image_1 to image_10
        img_path = os.path.join(images_folder, f'image_{i}.jpg')  # Update extension if needed
        if os.path.exists(img_path):
            images.append(img_path)
        else:
            images.append('https://via.placeholder.com/150')  # Placeholder if local image is not found
    return images

# Fetch the list of stock symbols
stocks = get_sp500_tickers()
stocks.insert(0, "Choose an option")

# New Feature: Additional Information boxes/containers on the right side
st.sidebar.markdown("<h1 style='font-size: 36px;'>Stock News</h1>", unsafe_allow_html=True)

# You can create multiple containers using Streamlit's columns
col1, col2 = st.sidebar.columns(2)

# Container 1 - Yahoo Finance News
with col1:
    st.sidebar.expander("Yahoo Finance", expanded=True)
    yahoo_news = fetch_yahoo_finance_news()
    images = get_local_images()
    for i, entry in enumerate(yahoo_news[:5]):  # Display the top 5 news articles
        st.sidebar.image(images[i], use_column_width=True)
        st.sidebar.write(f"**[{entry.title}]({entry.link})**")

# # Container 2 - Placeholder for other news
# with col2:
#     with st.sidebar.expander("Other News", expanded=True):
#         st.write("More news content goes here...")

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

    # Predict forecast with Prophet
    df_train = data[['Date', 'Close']]
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
