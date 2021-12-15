# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 12:34:41 2021

@author: Anders Huse
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

import seaborn as sns
import matplotlib.pyplot as plt

from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px

st.set_page_config(layout='wide')
### user input
#ticker = st.text_input('Enter company ticker:')

#%% Functions for use
def format_xaxes(figure: go.Figure, buttons_list: list, rs = True) -> go.Figure:
    """
    Formatting xaxes
    
    Parameters:
    -------------------------------------
    figure: Go.figure
        figure to have xaxes updated
        
    rs: Bool
        Whether or not to display the rangeslider
        
    buttons: list of int
        Integers representing buttons to be displayed

    
    returns: go.Figure()
        An updated plotly figure
        
    """
    figure.update_xaxes(
        rangeslider_visible=True if rs else False,
        rangeselector=dict(
            buttons=
            [
                dict(count=b['count'],
                     label=b['label'],
                     step=b['step'],
                     stepmode=b['stepmode']) for b in buttons_list
            ]
        ))
    
  

#%% Sidebar
ticker = st.sidebar.selectbox('Select company ticker', \
                              ('ELK', 'ORK', 'BRG', 'NOD', 'kahot'))
granularity = st.sidebar.selectbox('Select data granularity', \
                                   ('15m',  '1h', '1d', '5d', '1wk'))

### Getting long-name (Norwegian only)
df1 = yf.Ticker(ticker + '.OL')
long_name = df1.info['longName']


today = datetime.date.today()
lookback = datetime.timedelta(days=1000)
start = today - lookback

start_date = st.sidebar.date_input('Start date', start)
end_date = st.sidebar.date_input('End date', today)

if start_date < end_date :
    st.sidebar.success(f'Start date: {start_date}\n\nEnd date: {end_date}')
else:
    st.sidebar.error('Error: start date must preceed end date')
    
st.header(f'Dashborad for {long_name}')
    
#%% Reading data

#@st.cache()
def load_data(ticker: str, start: 'str', end: 'str', interval: 'str') -> pd.DataFrame:
    return yf.download(ticker + '.OL', start=start, end=end, interval=interval)

df = load_data(ticker, start_date, end_date, granularity)

#%% Defining df data
colors_bb = {'Upper band': '#ff207c',
             'Close': '#207cff',
             'Lower band': '#00ec8b'}
### df
indicator_df = BollingerBands(df.Close)

bb = df

bb['Upper band'] = indicator_df.bollinger_hband()
bb['Lower band'] = indicator_df.bollinger_lband()
bb = bb[['Close', 'Upper band', 'Lower band']]

#%% Metrics
returns = df.Close.pct_change()
c1, c2, c3, c4 = st.columns(4)

period_return = (df.Close[-1] - df.Close[0])/df.Close[0]

c1.metric('Return over period', f'{period_return:.2%}')
c2.metric('Mean return (daily)', f'{returns.mean():.3%}')
c3.metric('Standard deviation for', f'{returns.std():.3%}')
c4.metric('Number of observations', len(returns)) 
    
    
#%% Price andVolume (plotly)

buttons = [
    {'count':1, 'label':'1 month', 'step':'month', 'stepmode':'backward'},
    {'count':6, 'label':'6 months', 'step':'month', 'stepmode':'backward'},
    {'count':1, 'label':'1 year', 'step':'year', 'stepmode':'todate'},

    {'count':None, 'label':None, 'step':'all', 'stepmode':None}
]

st.subheader(f'{long_name} Price and Volume')

colors = {'red': '#ff207c',
          'grey': '#42535b',
          'blue': '#207cff',
          'orange': '#ffa320',
          'green': '#00ec8b'}

fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights = [0.7, 0.3],
                    # subplot_titles = ['chart_1', 'chart_2'],
                    vertical_spacing = 0.1
                   )

### Candlestick - OHLC
fig.add_trace(go.Candlestick(x     = df.index,
                             open  = df['Open'],
                             high  = df['High'],
                             low   = df['Low'],
                             close = df['Close'],
                             name  = 'OHLC'
                            ),row = 1, col = 1)
### Bollinger bands

fig.add_trace(go.Scatter(x = df.index,
                         y = df['Close'],
                         line = dict(color=colors_bb['Close'], width=.5),
                         name = 'Middle Band'))

fig.add_trace(go.Scatter(x = bb.index,
                         y = bb['Upper band'],
                         line = dict(color=colors_bb['Upper band'], width=1.0),
                         name = 'Upper Band (Sell)'))

fig.add_trace(go.Scatter(x = bb.index,
                         y = bb['Lower band'],
                         line = dict(color=colors_bb['Lower band'], width=0.7),
                         name = 'Lower Band (Buy)'))
### RSI
rsi = RSIIndicator(df.Close).rsi() 

### RSI
fig.add_trace(go.Bar(x=df.index,
                     y=df.Volume,
                     name = 'Volume',
                     marker_color = 'crimson',
                     ), row = 2, col = 1)

### Update visuals
fig.update_layout(
                  xaxis_rangeslider_visible = True,
       
                  yaxis_title = 'OHLC',
      
                  height=800,width=1200
                  )

### Updating xaxes
format_xaxes(fig, buttons, rs=False)

st.plotly_chart(fig)

#%% Display df
if st.checkbox('Display DataFrame:'):
    st.write(df.drop(['Upper band', 'Lower band', 'Adj Close'],axis=1))

#%%
# config_ticks = {'size': 14, 'color': colors['grey'], 'labelcolor': colors['grey']}
# config_ticks_2 = {'size': 20, 'color': colors['grey'], 'labelcolor': colors['grey']}
# config_title = {'size': 18, 'color': colors['grey'], 'ha': 'left', 'va': 'baseline'}

# ### Plotting Cose price and Volume in subplots

# plt.rc('figure', figsize=(16, 10))

# fig, axs = plt.subplots(2, 1, gridspec_kw = {'height_ratios': [3, 1]}, sharex=True)
# fig.tight_layout()

# axs[0].plot(df.index, df.Close, color=colors['blue'], linewidth=2, label='Close Price')
# axs[1].bar(df.index, df.Volume, width=3, color='blue')

# # axs[0] Visuals

# axs[0].yaxis.tick_right()
# axs[0].tick_params(axis='both', **config_ticks)
# axs[0].set_ylabel('Price - NOK')
# axs[0].yaxis.set_label_position('right')
# axs[0].yaxis.label.set_color(colors['grey'])
# axs[0].grid(axis='y', color = 'gainsboro', linestyle='-', linewidth=0.7)
# axs[0].set_axisbelow(True)

# plt.xticks(rotation=35, ha='right', fontsize=14, color = colors['grey'])

# # Spines
# axs[0].spines['top'].set_visible(False)
# axs[0].spines['left'].set_visible(False)

# axs[0].spines['right'].set_color(colors['grey'])
# axs[0].spines['bottom'].set_color(colors['grey'])

# axs[1].spines['top'].set_visible(False)
# axs[1].spines['left'].set_visible(False)

# axs[1].spines['right'].set_color(colors['grey'])
# axs[1].spines['bottom'].set_color(colors['grey'])


# ### Plotting additional Moving Averages

# mov_avg = {
#     'MA (50)': {'Range': 20, 'Color': colors['orange']}, 
#     'MA (100)': {'Range': 50, 'Color': colors['green']}, 
#     'MA (200)': {'Range': 100, 'Color': colors['red']}
# }

# for ma, ma_info in mov_avg.items():
#     axs[0].plot(df.index, df['Close'].rolling(ma_info['Range']).mean(),
#                color=ma_info['Color'], label=ma, linewidth=2, ls='--')

# # Legends
# axs[0].legend(loc='upper left', bbox_to_anchor= (-0.005, 0.95), fontsize=14);

# st.pyplot(fig)  

#%% st.columns

# c1, c2 = st.columns([3, 1])


# c1.subheader('Traded Volume: ')
# c1.bar_chart(df.Volume)

# c2.subheader('Latest 10 data points')

# vis_f = df.Close.to_frame()
# vis_f.set_index(vis_f.index.astype('str'), inplace=True)
# c2.write(vis_f[:-10])

#%% Volume

# st.subheader('Traded Volume: ')
# st.bar_chart(df.Volume)

#%% Technical Indicators

st.subheader('Technical Indicators')

### BB
if st.checkbox('Bollinger Bands: '):
    with st.container():
        
        sns.set_theme()
        fig =  sns.relplot(data    = bb,
    
    
                            kind   = 'line',
                            height = 5,
                            aspect = 2,
                            palette = colors_bb
                            
                           ).set(
                                 ylabel = 'Closeing Price',
                                 xlabel = None
                                )
        plt.tight_layout()
        plt.xticks(rotation=35)
        plt.show()
        
        st.pyplot(fig)
        
### MACD
macd = MACD(df.Close).macd()

if st.checkbox(f'Display MACD for {long_name}'):
    st.write(f'MACD Indicator for {long_name}')
    st.area_chart(macd)

### RSI
rsi = RSIIndicator(df.Close).rsi()

if st.checkbox(f'Display RSI for {long_name}'):
    st.write(f'RSI Indicator for {long_name}')
    st.line_chart(rsi) 

### STOCH
stoch = StochasticOscillator(high=df.High,
                             close=df.Close,
                             low=df.Low,
                             window=14,
                             smooth_window=3)
stoch_ = stoch.stoch()
stoch_signal = stoch.stoch_signal()

if st.checkbox('Stochastic Oscillator'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index,
                             y=stoch.stoch(),
                             name='STOCH',
                             line=dict(color='grey', width=1.5))
                            )
    
    fig.add_trace(go.Scatter(x=df.index,
                             y=stoch.stoch_signal(),
                             name='STOCH signal',
                             line=dict(color='magenta', width=1.5))
                           )
    ### Layout
    fig.update_layout(height=600, width=1200,
                      showlegend=True,
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)
