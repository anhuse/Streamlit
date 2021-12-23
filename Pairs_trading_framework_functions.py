#!/usr/bin/env python
# coding: utf-8

# # Pairs trading - EURONEXT sectors

# 1. [Functionality](#Functionality)
# 2. [Getting_tickers](#Getting_tickers)
# 3. [Making_close_df](#Making_close_df)
# 3. [Testing_for_colinearity](#Testing_for_colinearity)
# 4. [Calculating_spread_and_ratio](#Calculating_spread_and_ratio)
# 5. [z_scores](#z_scores)
# 7. [trading_signals](#trading_signals)
# 8. [Feature_engineering](#Feature_engineering)
# 9. [Assembling_the_model](#Assembling_the_model)

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns; sns.set(style="darkgrid")

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller


# ### Functionality

# In[5]:


def make_close_df(ticker_list, start, end):
    """
    Returns pandas DataFrame containing close values for all tickers in ticker_list, in the interval [start, end]
    
    Parameters
    -----------
    
    ticker_lis : list
        list of tickers
        
    start : str
        start date for time interval
        
    end : str
        end date for time interval
        
    Returns
    -----------
    DataFrame
        Pandas DataFrame containing close values
    """
    
    df = pd.DataFrame()
    
    for t in ticker_list:
        ticker = yf.Ticker(t)
        data = ticker.history(start=start, end=end)['Close']
        df[t] = data
        
    return df


# In[6]:


def find_cointegrated_pairs(data):
    """
    Function for finding cointegrated security pairs
    Returns multiple measures on the cointegration of several securities
    
    Parameters
    -----------
    
    data : pandas DataFrame 
        data frame of securities
    
    Returns
    -----------
    score_matrix : np.array
        matrix containing scores
        
    pvalue_matrix : np.array
        matrix containing pvalues
        
    pairs : list             
        list of cointegrated pairs of securities
    """
    
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


# In[20]:


def plot_coint_pairs(pvalues, ticker_list, cmap):
    """
    Function for plotting the cointegrated pairs in ticker_list
    
    Parameters
    -----------
    
    pvalues : numpy.ndarray
        cointegration p-values of securities in ticker_list
        
    ticker_list _ list
        list of securitites to incveastigate for possible cointegration
        
    cmap : str
        seaborn colormap for plotting purposes
   
    Returns
    -----------
    sns.heatmap
        plot visualising cointegratede pairs
    """
    
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(pvalues,
                    xticklabels = ticker_list,
                    yticklabels = ticker_list,
                    cmap        = cmap,
                    mask        = (pvalues >= 0.05)
                   )
    print(pairs)
    


# In[75]:


def calc_and_plot_spread(S1, S2, sec_1, sec_2, figsize=(12,6)):
    """
    Function for calculating and plotting the spread of two securitites
    
    Parameters
    -----------
    
    S1, S2 : pandas Series
        Series of price data (Close)
        
    Returns
    -----------
    matplotlib figure
        plot visualising spread of two securities
    """
    S1 = sm.add_constant(S1)
    results = sm.OLS(S2, S1).fit()
    S1 = S1[sec_1]
    b = results.params[sec_1]
    spread = S2 - b * S1
    
    spread.plot(figsize=figsize)
    plt.axhline(spread.mean(), color='black', linestyle = '--', alpha=0.8)
    plt.xlabel('')
    
    plt.tight_layout()
    # plt.title(f'Spread - {sec_1} and {sec_2}\n'.upper(), loc='left', size = 'x-large')
 


# In[76]:


def calculate_and_plot_ratio(S1, S2, figsize=(12,6)):
    """
    calculates the ratio of two securities
    
    Parameters
    -----------
    
    S1, S2 : pandas Series
        Series of price data (Close)
        
    start, end: string
        start / end of the time interval
    
    Returns
    -----------
    matplotlib figure
        visualisation of the ratio, including a horisontal mean-line 
    
    """
    
    ratio = S1 / S2
    ratio.plot(figsize=figsize)
    plt.axhline(ratio.mean(), color='black', linestyle = '--', alpha=0.8)
    plt.legend(['Price Ratio'])
    
    plt.xlabel('')
    plt.tight_layout()
#    plt.title(f'Ratio - {sec_1[:-3]} and {sec_2[:-3]}\n'.upper(), loc='left', size = 'x-large')
 
    


# In[107]:


def z_score(series):
    """
    Calculates z-scores for a given timeseries
    
    Parameters
    -----------
    series : Pandas timeseries
        time series
        
    Returns
    -----------
    Pandas timeseries
        z-scores of series
    """
    return (series - series.mean()) / np.std(series)


def plot_z_score(S1, S2, figsize=(12,6)):
    
    
    """
    Function for plotting z-score, including a lower and upper band s
    
    Parameters
    -----------
    
    S1, S2 : pandas Series
        Series of price data (Close)
        
    start, end: string
        start / end of the time interval
    
    Returns
    -----------
    matplotlib figure
        visualisation of the z-score, including a upper and lower band
    
    """
    
    ratio = S1 / S2
    z_score(ratio).plot(figsize=figsize, label = 'z-score')
    
    plt.axhline(z_score(ratio).mean(), color='k', linestyle = '-.', alpha=0.9, label='z-score mean')
    plt.axhline(1.0, color='red', linestyle = '--', alpha=0.9, label = 'upper border')
    plt.axhline(-1.0, color='green', linestyle = '--', alpha=0.9, label = ' lower border')
    
    plt.xlabel('')
    plt.legend()
    plt.tight_layout()
    plt.show()
   


# ### Getting_tickers

