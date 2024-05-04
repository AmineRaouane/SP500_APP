import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import plotly.graph_objects as go

st.title("SP_500 App 📊")

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
Tickers = sp500['Symbol'].unique().tolist()

ticker = st.selectbox("Choose a Ticker to analyse:",Tickers)

Data = yf.download(ticker)
Data.reset_index(inplace=True)
Data['Date'] = pd.to_datetime(Data['Date'])
Data['Date'] = Data['Date'].dt.tz_localize(None)
Data = Data[Data['Date'] > '2015-01-01']

if 'StartDate' not in st.session_state:
    st.session_state.StartDate = Data['Date'].iloc[0]
if 'EndDate' not in st.session_state:
    st.session_state.EndDate = Data['Date'].iloc[-1]

def Apply_button():
    st.session_state.StartDate = Start
    st.session_state.EndDate = End
def reset_Data():
    st.session_state.StartDate = Data['Date'].iloc[0]
    st.session_state.EndDate = Data['Date'].iloc[-1]

c1, c2 = st.columns(2)
with c1:
    Start = st.date_input("Start Date:", value=st.session_state.StartDate , min_value=Data['Date'].iloc[0], max_value=Data['Date'].iloc[-1]+pd.Timedelta(days=365))
    st.button('Reset', on_click=reset_Data, key="reset_button", use_container_width=True)
with c2:
    End = st.date_input("End Date:", value=st.session_state.EndDate, max_value=Data['Date'].iloc[-1])
    st.button('Apply', on_click=Apply_button, key="apply_button", use_container_width=True)

Filtered_Data = Data[(Data['Date'] >= pd.Timestamp(st.session_state.StartDate)) & (Data['Date'] <= pd.Timestamp(st.session_state.EndDate))]

# fig, ax = mpf.plot(Filtered_Data.set_index("Date"), type='candle', style='charles', title='Candlestick Chart',ylabel='Price (USD)',figratio=(20,6),returnfig=True)
candles = go.Candlestick(
    x=Filtered_Data['Date'],
    open=Filtered_Data['Open'],
    high=Filtered_Data['High'],
    low=Filtered_Data['Low'],
    close=Filtered_Data['Close']
)

layout = {
    'xaxis': {'title': 'Date'},
    'yaxis': {'title': 'Price'}
}

figure = go.Figure(data=[candles], layout=layout)

st.plotly_chart(figure)