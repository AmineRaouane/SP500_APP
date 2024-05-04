import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import pandas_ta as ta

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("SP_500 App 📊")

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
Tickers = sp500['Symbol'].unique().tolist()
ticker = st.selectbox("Select Ticker to start :",Tickers)

if ticker :
    Data = yf.download(ticker)
    Data.reset_index(inplace=True)
    Data['Date'] = pd.to_datetime(Data['Date']) # type: ignore
    Data['Date'] = Data['Date'].dt.tz_localize(None)
    Data = Data[Data['Date'] > '2020-01-01']

    Indicators = ['Stochastic Oscillator','Bollinger Bands',
              'Moving Average Convergence Divergence','Dollar Volume',
              'Relative Strength Index','Moving Average Crossover',
              'Garman-Klass Volatility','Average True Range']
    Indicator = st.selectbox("Choose an indicator:",Indicators)

if Indicator == 'Moving Average Crossover' :
    c1, c2 = st.columns(2)
    with c1:
        MA_min = st.slider('Enter the short_term moving average', min_value=3, max_value=10, value=7)
    with c2:
        MA_max = st.slider('Enter the long_term moving average', min_value=14, max_value=30, value=21)
    
    Data[f'{MA_min}-day'] = Data['Close'].rolling(MA_min).mean()
    Data[f'{MA_max}-day'] = Data['Close'].rolling(MA_max).mean()
    Data['signal'] = np.where(Data[f'{MA_min}-day'] > Data[f'{MA_max}-day'], 1, 0)
    Data['signal'] = np.where(Data[f'{MA_min}-day'] < Data[f'{MA_max}-day'], -1, Data['signal'])
    Data['entry'] = Data.signal.diff()
    def signals(data):
        buy_signals = [np.nan] * len(data) 
        sell_signals = [np.nan] * len(data)
        a= 0
        for i in data.index:
            if data.loc[i, 'entry'] == 2: 
                buy_signals[a] = data.loc[i, 'Close']
            elif data.loc[i, 'entry'] == -2:
                sell_signals[a] = data.loc[i, 'Close']
            a += 1
        return buy_signals, sell_signals
    buy_signals ,sell_signals = signals(Data)
    fig = mpf.plot(Data.set_index("Date"),type='candle', style='yahoo', title='Candlestick Chart',ylabel='Price (USD)', mav=(MA_min,MA_max), figratio=(20,6),
         addplot=[
             mpf.make_addplot(buy_signals, type='scatter', marker='^', markersize=100, color='g'),
             mpf.make_addplot(sell_signals, type='scatter', marker='v', markersize=100, color='r')
         ])
    st.pyplot(fig)
elif Indicator == 'Relative Strength Index':
    rsi_period = st.slider('Enter the RSI period', min_value=7, max_value=30, value=14)
    RSI = ta.rsi(Data["Close"], length=rsi_period)
    fig = mpf.plot(Data.set_index("Date"),type='candle', style='yahoo', title='Candlestick Chart', ylabel='Price (USD)',figratio=(20,6),
         addplot=[
                mpf.make_addplot(RSI, panel=1, color='b', title='RSI')
            ])
    st.pyplot(fig)
elif Indicator == 'Moving Average Convergence Divergence':
    c1, c2, c3 = st.columns(3)
    with c1:
        MACD_fast = st.slider('Enter the fast moving average', min_value=7, max_value=12, value=7)
    with c2:
        MACD_slow = st.slider('Enter the slow moving average', min_value=15, max_value=30, value=21)
    with c3:
        MACD_signal = st.slider('Enter the signal', min_value=7, max_value=14, value=9)

    MACD = ta.macd(Data["Close"], fast=MACD_fast, slow=MACD_slow, signal=MACD_signal)
    MACD_fast_array = MACD.iloc[:,0] # type: ignore
    MACD_slow_array = MACD.iloc[:,1] # type: ignore
    MACD_signal_array = MACD.iloc[:,2] # type: ignore
    fig = mpf.plot(Data.set_index("Date"),type='candle', style='yahoo', title='Candlestick Chart',ylabel='Price (USD)', mav=(7, 21), figratio=(20, 6),
         addplot=[mpf.make_addplot(MACD_fast_array, panel=1, color='blue'),
                  mpf.make_addplot(MACD_slow_array, panel=1, color='orange'),
                  mpf.make_addplot(MACD_signal_array, panel=1, color='green')])
    st.pyplot(fig)
elif Indicator == 'Bollinger Bands':
    Period = st.slider('Enter the window period', min_value=7, max_value=30, value=7)
    Middle_Band= Data["Close"].rolling(window=Period).mean()
    Upper_Band = Middle_Band + 2*Data["Close"].rolling(window=Period).std()
    Lower_Band = Middle_Band - 2*Data["Close"].rolling(window=Period).std()
    fig = mpf.plot(Data.set_index("Date"),type='candle', style='yahoo', title='Candlestick Chart', ylabel='Price (USD)', figratio=(20,6),mav=(Period),
            addplot=[
                mpf.make_addplot(Upper_Band, panel=0, color='g'),
                mpf.make_addplot(Lower_Band, panel=0, color='r')
            ])
    st.pyplot(fig)
elif Indicator == 'Stochastic Oscillator':
    Period = st.slider('Enter the window period', min_value=7, max_value=30, value=7)
    Oscillator =ta.stoch(Data["High"], Data["Low"], Data["Close"], window=Period) # type: ignore
    Data["STOCHk"] = Oscillator.iloc[:,0] # type: ignore
    Data["STOCHd"] = Oscillator.iloc[:,1] # type: ignore
    fig = mpf.plot(Data.set_index("Date"),type='candle', style='yahoo', title='Candlestick Chart', ylabel='Price (USD)', figratio=(20,6),
            addplot=[
                mpf.make_addplot(Data["STOCHk"], panel=1, color='b'),
                mpf.make_addplot(Data["STOCHd"], panel=1, color='g')
            ])
    st.pyplot(fig)
elif Indicator == 'Garman-Klass Volatility':
    garman_klass_vol = ((np.log(Data['High'])-np.log(Data['Low']))**2)/2-(2*np.log(2)-1)*((np.log(Data['Adj Close'])-np.log(Data['Open']))**2)
    fig = mpf.plot(Data.set_index("Date"),type='candle', style='yahoo', title='Candlestick Chart', ylabel='Price (USD)', figratio=(20,6),
            addplot=[
                mpf.make_addplot(garman_klass_vol, panel=1, color='b')
            ])
    st.pyplot(fig)
elif Indicator == 'Average True Range':
    Period = st.slider('Enter the window period', min_value=7, max_value=30, value=7)
    atr = ta.atr(high=Data['High'],low=Data['Low'],close=Data['Close'],length=Period)
    fig = mpf.plot(Data.set_index("Date"),type='candle', style='yahoo', title='Candlestick Chart', ylabel='Price (USD)', figratio=(20,6),
            addplot=[
                mpf.make_addplot(atr, panel=1, color='b')
            ])
    st.pyplot(fig)
elif Indicator == 'Dollar Volume':
    Dollar_Volume = (Data['Adj Close']*Data['Volume'])/1e6
    fig = mpf.plot(Data.set_index("Date"),type='candle', style='yahoo', title='Candlestick Chart', ylabel='Price (USD)', figratio=(20,6),
            addplot=[
                mpf.make_addplot(Dollar_Volume, panel=1, color='b')
            ])
    st.pyplot(fig)