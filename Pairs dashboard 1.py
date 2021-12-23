# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import statsmodels
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objs as go

from statsmodels.tsa.stattools import coint, adfuller

### Importing own functionality
from Pairs_trading_framework_functions import calc_and_plot_spread,\
    calculate_and_plot_ratio, z_score, plot_z_score

st.set_page_config(layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

#%% Functions

@st.cache()
def load_data(ticker: str, start: 'str', end: 'str') -> pd.DataFrame:
    return yf.download(ticker, start=start, end=end)

@st.cache
def make_close_df(df1, df2):
    df_close = pd.concat([df_1, df_2], axis=1).Open
    df_close.columns = [long_name1, long_name2]
    df_close.dropna(how='any', axis=0, inplace=True)
    return df_close


#%% Sidebar
ticker1 = st.sidebar.selectbox('Select ticker of the first company:', \
                              ('ELK.ol', 'ORK.ol', 'EQNR.ol'))

ticker2 = st.sidebar.selectbox('Select foreign ticker:', \
                              ('WCH.DE', 'BN.PA', 'BP'))

### Getting long-name
df1 = yf.Ticker(ticker1)
df2 = yf.Ticker(ticker2)

long_name1 = df1.info['longName']
long_name2 = df2.info['longName']


today = datetime.date.today()
lookback = datetime.timedelta(days=1000)
start = today - lookback

start_date = st.sidebar.date_input('Start date', start)
end_date = st.sidebar.date_input('End date', today)

if start_date < end_date :
    st.sidebar.success(f'Start date: {start_date}\n\nEnd date: {end_date}')
else:
    st.sidebar.error('Error: start date must preceed end date')
    
st.header(f'Dashborad for {long_name1} and {long_name2}')


#%% Making Dfs


df_1 = load_data(ticker1, start_date, end_date)
df_2 = load_data(ticker2, start_date, end_date)

df_close = make_close_df(df1, df2)

#%% p-value
S1 = df_close[long_name1]
S2 = df_close[long_name2]

score, pvalue, _ = coint(S1, S2)
print(pvalue)

if pvalue <= 0.05:
    delta = 'significant'
    delta_color = 'normal'
else:
    delta = 'non-significant'
    delta_color = 'inverse'

c1, c2 = st.columns(2)
c1.metric('p-value for chosen pair returns', f'{pvalue:.5f}', delta,
          delta_color)


#%% Main plot
st.subheader('Comparison of price deveelopment')
fig, ax = plt.subplots()
df_close[[long_name1, long_name2]].plot(figsize=(16,7), ax=ax)
plt.xlabel('')
#plt.title('Comparison of price deveelopment for chosen pair', fontsize=18,loc='left')
plt.show()
st.pyplot(fig)


#%% Display data
if st.checkbox(f'Dispaly data for {long_name1} & {long_name2}'):
    c1, c2 = st.columns(2)
    
    c1.subheader(f'Data for {long_name1}')
    c2.subheader(f'Data for {long_name2}')
    
    c1.write(df_1.head(10).drop('Adj Close', axis=1))
    c2.write(df_2.head(10).drop('Adj Close', axis=1))
    
#%%
st.subheader('Spread of pair prices')
fig, ax = plt.subplots()
ax = calc_and_plot_spread(df_close.iloc[:,0], df_close.iloc[:,1],
                          long_name1, long_name2, figsize=(16,7))
st.pyplot(fig)



#%%
st.subheader('The ratio of the two companies')
fig, ax = plt.subplots()
ax = calculate_and_plot_ratio(df_close.iloc[:,0], df_close.iloc[:,1],figsize=(16,7))
st.pyplot(fig)

#%%
st.subheader('z-score including lower and upper bounds')
fig, ax = plt.subplots()
ax = plot_z_score(df_close.iloc[:,0], df_close.iloc[:,1],figsize=(16,7))
st.pyplot(fig)