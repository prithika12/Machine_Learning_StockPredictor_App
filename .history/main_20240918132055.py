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
from bs4 import BeautifulSoup

START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set page configuration
st.set_page_config(layout="wide")

# Apply custom CSS
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

# Function to extract images from a news article
def extract_image_from_article(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tag = soup.find('img')
        img_url = img_tag['src'] if img_tag else None
        return img_url
    except Exception as e:
        print(f"Error fetching image from {url}: {e}")
        return None

# Fetch stock symbols
stocks = get_sp500_tickers()
stocks.insert(0, "Choose an option")

# Sidebar for stock news
st.sidebar.markdown("<h1 style='font-size: 36px;'>Stock News</h1>", unsafe_allow_html=True)

# Container for Yahoo Finance News
with st.sidebar.expander("Yahoo Finance", expanded=True):
    yahoo_news = fetch_yahoo_finance_news()
    for entry in yahoo_news[:5]:  # Display the top 5 news articles
        img_url = extract_image_from_article(entry.link)
        if img_url:
            st.sidebar.write(f"Image URL: {img_url}")  # Debug: Check the image URL
            try:
                st.sidebar.image(img_url, use_column_width=True)
            except Exception as e:
                st.sidebar.write(f"Error loading image: {e}")
                st.sidebar.image("https://via.placeholder.com/150", use_column_width=True)  # Placeholder image
        else:
            st.sidebar.image("https://via.placeholder.com/150", use_column_width=True)  # Placeholder image
        st.sidebar.write(f"**[{entry.title}]({entry.link})**")
        st.sidebar.write(entry.get('description', 'No description available'))

st.title("Stock Prediction App")

# Dropdown for choosing a stock symbol
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
