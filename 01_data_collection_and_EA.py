import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.patches as mpatches
from matplotlib.dates import date2num, DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
from mplfinance.original_flavor import candlestick_ohlc
import seaborn as sns
import streamlit as st
import datetime
from datetime import date, timedelta
from io import StringIO
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%matplotlib inline

st.title('FTSE_100 Tech analysis')

st.markdown("<hr>", unsafe_allow_html=True)

st.write('This  an excercise in technical analysis wrapped up in a catchy web-app form (thanks to Streamlit). The present page is step #1 in a 4-step analysis pipeline')

st.header('Data collection')

st.write('Historical data on several FTSE 100 index companies will be collected, analyzed, and visualized in an attempt to gain insights into their equity market performance from 2014 to 2024. The market behavior of the index itself will also be analyzed.')
text = """
A relatively safe, however representative list of stocks has been identified (huge thanks to Alison Mitchell):

- ULVR.L (Unilever)
- SHEL.L (Shell)
- GSK.L (GlaxoSmithKline)
- AZN.L (AstraZeneca)
- HSBA.L (HSBC)
- BP.L (BP)

This list represents a selection of different industries, namely - pharmaceuticals, oil, and finance.
"""
st.write(text)

ftse100_stocks = yf.download("AZN.L GSK.L ULVR.L BP.L SHEL.L HSBA.L", start=datetime.datetime(2014, 1, 1), 
                                     end=datetime.datetime(2023, 12, 31), group_by='tickers')
ftse100_stocks.head(10)
st.dataframe(ftse100_stocks.head(10))
#------------------------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.write('Let us attempt some data exploration')
ftse100_stocks.describe()
description = ftse100_stocks.describe()
st.dataframe(description)
#------------------------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.write('Lets summarise the data to the dataframe to see if any values of datatypes are missing')
buffer = StringIO()
ftse100_stocks.info(buf=buffer)
info_str = buffer.getvalue()

# Displaying the info string using markdown for better formatting
st.markdown("```\n" + info_str + "\n```")
#------------------------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.write('Number of rows represents the number of trading days')
ftse100_stocks.shape
st.dataframe(ftse100_stocks.shape)
#------------------------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.write('Adjusted Close price for each company stock.') 

adj_close = pd.DataFrame()

tickers = ['AZN.L', 'GSK.L', 'ULVR.L', 'BP.L', 'SHEL.L', 'HSBA.L']
for ticker in tickers:
    adj_close[ticker] = ftse100_stocks[ticker]['Adj Close']

adj_close

adj_close.plot(grid = True)
sns.set(rc={'figure.figsize':(15, 9)})
plt.title('Adjusted Close Price for all stocks', color = 'black', fontsize = 20)
plt.xlabel('Year', color = 'black', fontsize = 15)
plt.ylabel('Adjusted Close Price (pence)', color = 'black', fontsize = 15);

st.write('Lets alculate min and max Adjusted Close price')

adj_close_min_max = adj_close.apply(lambda x: pd.Series([x.min(), x.max()], 
                              index=['min', 'max']))

adj_close_min_max

# Plot BP.L and HSBA.L data on a secondary y-axis

adj_close.plot(secondary_y = ["BP.L", "HSBA.L"], grid = True)
sns.set(rc={'figure.figsize':(15, 9)})
plt.title('Adjusted Close Price with two different scales', color = 'black', fontsize = 20);

# So we would want to plot return_{t,0}  = \frac{price_t}{price_0} by applying the lambda function to each column in an adjusted close datatframe.
returns_lambda = adj_close.apply(lambda x: x / x[0])
returns_lambda.head()

# Plot return_{t,0}  = \frac{price_t}{price_0} with transformed data to get an insight on how profitable the stock had been.

returns_lambda.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
sns.set(rc={'figure.figsize':(15, 9)})
plt.title('Stock returns for 10 year time period', color = 'black', fontsize = 20)
plt.xlabel('Year', color = 'black', fontsize = 15)
plt.ylabel('Returns (%)', color = 'black', fontsize = 15);

# Create dataframe to contain returns for each company's stock to plot the change of each stock per day

returns = pd.DataFrame()

# This can be achieved with the pandas  pct_change() method which computes the percentage change from the previous row by default.

tickers = ['AZN.L', 'GSK.L', 'ULVR.L', 'BP.L', 'SHEL.L', 'HSBA.L']
for ticker in tickers:
    returns[ticker] = ftse100_stocks[ticker]['Adj Close'].pct_change() * 100

returns

# Cleaning the data by dropping the NaN values

returns.dropna(inplace=True)
returns.head()

# Plot returns for 2023 that will show changes between trading days. THis is generally considered as a more advanced approach to modelling of equity behaviour.

returns.loc['2023-01-01':'2023-12-31'].plot(grid = True).axhline(y = 1, color = "black", lw = 2)
sns.set(rc={'figure.figsize':(15, 9)})
plt.title('Stock returns for 2023', color = 'black', fontsize = 20)
plt.xlabel('Date', color = 'black', fontsize = 15)
plt.ylabel('Returns (%)', color = 'black', fontsize = 15);

# Use numpy's log function to obtain and plot the log differences of the adjusted price data

stock_change = adj_close.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.

stock_change.head()

# Clean up the data by dropping NaNs

stock_change.dropna(inplace=True)
stock_change.head()

# Plot log differences for 2014-2024

stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
sns.set(rc={'figure.figsize':(15, 9)})
plt.title('Log differences of stocks for 10 year time period', color = 'black', fontsize = 20)
plt.xlabel('Year', color = 'black', fontsize = 15)
plt.ylabel('Natural log', color = 'black', fontsize = 15);

# Plot log differences for 2023 (since 2024 has just begun)

stock_change.loc['2023-01-01':'2023-12-31'][1:].plot(grid = True).axhline(y = 0, color = "black", lw = 2)
sns.set(rc={'figure.figsize':(15, 9)})
plt.title('Log differences of stocks for 2023', color = 'black', fontsize = 20)
plt.xlabel('Year', color = 'black', fontsize = 15)
plt.ylabel('Natural log', color = 'black', fontsize = 15);

# To keep the returns on the same time scale - the annual percentage rate needs to be computed
stock_change_apr = stock_change * 252 * 100    # There are 252 trading days in a year; the 100 converts to percentages

stock_change_apr

# Plotting annualised returns for the last year (2023)

stock_change_apr['2023-01-01':'2023-12-31'].plot(grid = True).axhline(y = 0, color = "black", lw = 2)
sns.set(rc={'figure.figsize':(15, 9)})
plt.title('Annual percentage rate (APR) for 2023', color = 'black', fontsize = 20)
plt.xlabel('Date', color = 'black', fontsize = 15)
plt.ylabel('APR', color = 'black', fontsize = 15);

# Worst single day returns

returns.idxmin()

# Best single day returns

returns.idxmax()


# Computing mean to give a representation of the average expected returns 

returns.mean()

# Computing variance to give a measure of the dispersion of returns around the mean

returns.var()

# Computing the standard deviation to describe variability in the stock returns from the mean 
 
returns.std()

# Computing skewness to measure the asymmetry of the data around its mean

returns.skew()

# Computing kurtosis as a measure of the combined sizes of the two tails.

returns.kurt()

# Pairplot of returns dataframe 

sns.pairplot(returns);

# Boxplots showing distribution of the returns data over the time period 

sns.set_style("whitegrid")
fig, axs = plt.subplots(ncols=6, nrows=1, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in returns.items():
    sns.boxplot(y=k, data=returns, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# Distribution plots showing the data for returns for 2023 

sns.set_style("white")

tickers = ['AZN.L', 'GSK.L', 'ULVR.L', 'BP.L', 'SHEL.L', 'HSBA.L']
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()

for ticker in tickers:
    sns.histplot(returns.loc['2023-01-01':'2023-12-31'][ticker], color='green', bins=100, ax=axs[index], kde=True)
    index += 1


# Covariance matrix to show direction of relationship between stocks' returns

returns.cov() 

# Correlation matrix to show strength and direction of relationship between stocks' returns

returns.corr()

# The heatmap clearly shows the strength of correlation between pairs of company returns

plt.figure(figsize=(10, 10))
sns.heatmap(data = returns.corr(), vmax=.8, linewidths=0.5,  fmt='.2f',
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.show()


# Download FTSE 100 historical stock data from Yahoo! Finance for 1984-2020

ftse100_idx_to_2024 = yf.download("^FTSE", start=datetime.datetime(1984, 1, 1), 
                                     end=datetime.datetime(2024, 1, 1))
ftse100_idx_to_2024

# Now let's visualize the data
def ftse100_to_2024_plot():
    ftse100_idx_to_2024['Close'].plot(grid = True)
    sns.set(rc={'figure.figsize':(20, 10)})
    plt.axvspan('1987','1989',color='r',alpha=.5)
    plt.axvspan('2008','2010',color='r',alpha=.5)
    plt.axvspan('2020','2024',color='r',alpha=.5)
    labs = mpatches.Patch(color='red',alpha=.5, label="Black Monday, 2008 Crash, Covid-19 fall and its aftermath")
    plt.legend(handles=[labs], prop={"size":15},  bbox_to_anchor=(0.4, 0.1), loc='upper center', borderaxespad=0.)
    plt.title('Close Price for FTSE 100 stocks', color = 'black', fontsize = 20)
    plt.xlabel('Year', color = 'black', fontsize = 15)
    plt.ylabel('Close Price (pence)', color = 'black', fontsize = 15)
    plt.show();

ftse100_to_2024_plot()


