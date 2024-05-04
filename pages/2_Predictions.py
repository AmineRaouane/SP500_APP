import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet #type: ignore
from prophet.diagnostics import cross_validation #type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX #type: ignore

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("SP_500 App 📊")

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
Tickers = sp500['Symbol'].unique().tolist()
ticker = st.selectbox("Select Ticker to start :",Tickers)
Period = st.slider('Enter the period desired for forcasting', min_value=7, max_value=45, value=14)

def Prophet_pipeline(Data,periods):
    Data = Data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    model = Prophet()
    model.fit(Data)
    # df_cv = cross_validation(model, initial=f'{len(Data)//2} days', period='30 days', horizon = '30 days')
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[-periods:]

def Arima_pipeline(Data,periods,order=(1,2,1),seasonal_order = (1,2,1,12)):
    Data = Data[['Date', 'Close']]
    Data.set_index('Date', inplace=True)

    model = SARIMAX(Data, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=periods) # type: ignore
    return forecast

if ticker :
    Data = yf.download(ticker)
    Data.reset_index(inplace=True)
    Data['Date'] = pd.to_datetime(Data['Date']) # type: ignore
    Data['Date'] = Data['Date'].dt.tz_localize(None)
    Data = Data[Data['Date'] > '2020-01-01']


if st.columns(3)[1].button('predict',use_container_width=True):
    ARIMA_predictions = Arima_pipeline(Data,Period) 
    Prophet_predictions = Prophet_pipeline(Data,Period)["yhat"] # type: ignore
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(Data.iloc[-30:]['Close'].values,label="Last 30 days")
    ax.plot(range(30+Period),np.concatenate([np.nan * np.ones(29) , [Data.iloc[-1]['Close']] ,(Prophet_predictions+ARIMA_predictions)/2]),label="ARIMA/Prophet_predictions")
    # ax.plot(np.concatenate([Data.iloc[-30:]['Close'].values, ARIMA_predictions]),label="ARIMA_predictions")
    # ax.plot(np.concatenate([Data.iloc[-30:]['Close'].values, Prophet_predictions]),label="Prophet_predictions")
    ax.legend()
    st.pyplot(fig)