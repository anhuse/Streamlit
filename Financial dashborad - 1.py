# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 10:24:33 2021

@author: Anders Huse
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

import seaborn as sns

from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator

### user input
#ticker = st.text_input('Enter company ticker:')

#%% Sidebar
ticker = st.sidebar.selectbox('Select company ticker', \
                              ('ELK', 'ORK', 'BRG', 'NOD'))

today = datetime.date.today()
lookback = datetime.timedelta(days=1000)
start = today - lookback

start_date = st.sidebar.date_input('Start date', start)
end_date = st.sidebar.date_input('End date', today)

if start_date < end_date :
    st.sidebar.success(f'Start date: {start_date}\n\nEnd date: {end_date}')
else:
    st.sidebar.error('Error: start date must preceed end date')
    

    

#%% Reading data

@st.cache()
def load_data(ticker: str, start: 'str', end: 'str') -> pd.DataFrame:
    return yf.download(ticker + '.OL', start=start, end=end)

df = load_data(ticker, start_date, end_date)

#%% Returns
df_ret = df.Close.pct_change()
#%% Title and div facts
st.title(f'Dashborad for {ticker}')
st.markdown('This dashborad gives an overview of the chosen\
            security, giving the visitor insight in price development \
            traded volume and technical analysis.')


#%% Visualizing stats and plotting
df_viz = df[['Open', 'Close', 'High', 'Low', 'Volume']]
st.subheader('Recent data (last 10 days):')
st.write(df_viz.tail(10))

st.subheader(f'Price development - {ticker}')
st.line_chart(df.Close)

st.subheader('Traded Volume: ')
st.bar_chart(df.Volume)

st.subheader('Descrtptive stats')
st.write(df_viz.describe().T)


#%% Technical Indicators

### BB
indicator_bb = BollingerBands(df.Close)

bb = df

bb['Upper band'] = indicator_bb.bollinger_hband()
bb['Lower band'] = indicator_bb.bollinger_lband()
bb = bb[['Close', 'Upper band', 'Lower band']]

### MACD
macd = MACD(df.Close).macd()

### RSI
rsi = RSIIndicator(df.Close).rsi()   

#%% Plot prices and bollinger bands
st.subheader(f'Technical Indicators for {ticker}')
### BB
if st.checkbox(f'Display Bollinger Bands for {ticker}'):
    st.write(f'Bollinger bands: {ticker}')
    st.line_chart(bb)


### MACD
if st.checkbox(f'Display MACD for {ticker}'):
    st.write(f'MACD Indicator for {ticker}')
    st.area_chart(macd)

### RSI
if st.checkbox(f'Display RSI for {ticker}'):
    st.write(f'RSI Indicator for {ticker}')
    st.line_chart(rsi)
    

#%% Seaborn plots
st.subheader('Seaborn visualisations')


if st.checkbox("Seaborn Pairplot", value=True):
	fig = sns.pairplot(data=df_ret.to_frame(),
                       height = 5,
                       aspect = 2).set(
                                       ylabel = 'Returns',
                                       xlabel = None
                    )
	st.pyplot(fig)
    
if st.checkbox("Seaborn Relplot", value=True):
    sns.set_theme()
    fig =  sns.relplot(data   = df,
                x      = df.index,
                y      = 'Close',
                kind   = 'line',
                height = 5,
                aspect = 2   # Adjusts the width over height ratio
                
               ).set(
                     ylabel = 'Closeing Price',
                     xlabel = None
                    )
    st.pyplot(fig)
        
    
    
     
    
    
    
    
    
    
    
    
    
    
    
    