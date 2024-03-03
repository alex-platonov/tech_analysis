import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.dates import date2num, DateFormatter, WeekdayLocator, DayLocator, MONDAY
import seaborn as sns
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
from scipy import stats
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
import datetime
from datetime import date, timedelta
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

st.title('FTSE_100 Technical analysis')
st.subheader('Step 2: Patterns and Indicators')
st.image('https://raw.githubusercontent.com/alex-platonov/tech_analysis/main/02_patterns_and_indicators.jpg')

st.markdown("<hr>", unsafe_allow_html=True)

st.subheader('Introduction')
st.write('Technical analysis is the use of charts and technical indicators to identify trading signals and price patterns. Various technical strategies will be investigated using the most common indicators.')

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.write('Let us begin again by downloading FTSE 100 stock data from Yahoo! Finance and storing it in a pandas dataframe. The same constituent companies as in Step - 1 will be selected.')

ftse100_stocks = yf.download("AZN.L GSK.L ULVR.L BP.L SHEL.L HSBA.L", start=datetime.datetime(2014, 1, 1), 
                                     end=datetime.datetime(2023, 12, 31), group_by='tickers')
st.dataframe(ftse100_stocks.head(10))
#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.subheader('The Honorable Guinea Pig')
st.write('HSBC stock (HSBA.L) will be selected for plotting charts and testing various trading strategies for no specific reason other than personal preference.')

hsba = ftse100_stocks['HSBA.L']
                                     
st.dataframe(hsba.head())
#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.subheader('Visualising stock data')
st.write('For some initial visualization we are going to use a Japanese candlestick chart. Such carts are widely used in a particular trading style called price action to predict market movement through pattern recognition of continuations, breakouts, and reversals. Unlike a line chart, all of the price information can be viewed in one figure that shows the high, low, open, and close price of the day or chosen time frame. Price action traders observe patterns formed by green bullish candles where the stock is trending upwards over time, and red bearish candles where there is a downward trend.')
st.set_option('deprecation.showPyplotGlobalUse', False)
def pandas_candlestick_ohlc(dat, stick="day", otherseries=None, txt=""):
    """
    Japanese candlestick chart showing OHLC prices for a specified time period

    :param dat: pandas DataFrame object with datetime64 index, and float columns "Open", "High", "Low", and "Close"
    :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs include "day", "week", "month", and "year", ("day" default), and any numeric input indicates the number of trading days included in a period
    :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines
    :param txt: Title text for the candlestick chart

    :returns: a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
    """
    sns.set(rc={'figure.figsize':(20, 10)})
    sns.set_style("whitegrid")  # Apply seaborn whitegrid style to the plots 

    transdat = dat.loc[:, ["Open", "High", "Low", "Close"]].copy()

    if type(stick) == str and stick in ["day", "week", "month", "year"]:
        if stick != "day":
            if stick == "week":
                transdat['period'] = pd.to_datetime(transdat.index).map(lambda x: x.strftime('%Y-%U'))
            elif stick == "month":
                transdat['period'] = pd.to_datetime(transdat.index).map(lambda x: x.strftime('%Y-%m'))
            elif stick == "year":
                transdat['period'] = pd.to_datetime(transdat.index).map(lambda x: x.strftime('%Y'))
            
            grouped = transdat.groupby('period')
            plotdat = pd.DataFrame([{
                "Open": group.iloc[0]["Open"],
                "High": max(group["High"]),
                "Low": min(group["Low"]),
                "Close": group.iloc[-1]["Close"]
            } for _, group in grouped], index=pd.to_datetime([period for period, _ in grouped]))
        else:
            plotdat = transdat
            plotdat['period'] = pd.to_datetime(plotdat.index)
    elif type(stick) == int and stick >= 1:
        transdat['period'] = np.floor(np.arange(len(transdat)) / stick)
        grouped = transdat.groupby('period')
        plotdat = pd.DataFrame([{
            "Open": group.iloc[0]["Open"],
            "High": max(group["High"]),
            "Low": min(group["Low"]),
            "Close": group.iloc[-1]["Close"]
        } for _, group in grouped], index=[group.index[0] for _, group in grouped])
    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')

    plotdat['date_num'] = date2num(plotdat.index.to_pydatetime())

    fig, ax = plt.subplots()
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    sns.set(rc={'figure.figsize':(20, 10)})
    candlestick_ohlc(ax, plotdat[['date_num', 'Open', 'High', 'Low', 'Close']].values, width=0.6/(24*60), colorup='g', colordown='r')

    if otherseries is not None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        for series in otherseries:
            dat[series].plot(ax=ax, lw=1.3)

    plt.title(f"Candlestick chart of HSBA.L OHLC stock prices from 01 Jan 2014 - 31 Dec 2023", color = 'black', fontsize = 20)
    plt.xlabel('Date', color = 'black', fontsize = 15)
    plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15)
    candlestick_ohlc(ax, plotdat[['date_num', 'Open', 'High', 'Low', 'Close']].values, width=20, colorup='g', colordown='r')

    plt.show()

st.pyplot(pandas_candlestick_ohlc(hsba, stick="month"))
#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.header('Technical Indicators and Strategies')

st.write('Let us begin with a textbook definition: A technical indicator is a series of data points that are derived by applying a formula to the price data of a security. They are price-derived indicators that use formulas to translate the momentum or price levels into quantifiable time series. There are two categories of indicators: leading and lagging, and four types: trend, momentum, volatility, and volume, which serve three broad functions: to alert, to confirm, and to predict.')

st.subheader('Trend-following strategies')
st.write('Trend-following is about profiting from the prevailing trend through buying an asset when its price trend goes up, and selling when its trend goes down, expecting price movements to continue.')

st.subheader('Moving averages')
st.write('Moving averages smooth a series filtering out the noise to help identify trends, one of the fundamental principles of technical analysis being that prices move in trends. Types of moving averages include simple, exponential, smoothed, linear-weighted, MACD, and lagging indicators they follow the price action and are commonly referred to as trend-following indicators.')

st.subheader('Simple Moving Average (SMA)')
st.write('The simplest form of a moving average, known as a Simple Moving Average (SMA), is calculated by taking the arithmetic mean of a given set of values over a set time period. This model is probably the most naive approach to time series modeling and simply states that the next observation is the mean of all past observations and each value in the time period carries equal weight.')

st.write('Modelling this an as average calculation problem we would try to predict the future stock market prices (for example, xt+1 ) as an average of the previously observed stock market prices within a fixed size window (for example, xt-n, ..., xt). This helps smooth out the price data by creating a constantly updated average price so that the impacts of random, short-term fluctuations on the price of a stock over a specified time frame are mitigated.')

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

def sma():
  plt.figure(figsize=(15,9))
  ftse100_stocks[ticker]['Adj Close'].loc['2023-01-01':'2023-12-31'].rolling(window=20).mean().plot(label='20 Day Avg')
  ftse100_stocks[ticker]['Adj Close'].loc['2023-01-01':'2023-12-31'].plot(label=f"{label_txt}")
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15);
  plt.legend()

ticker = 'HSBA.L'
title_txt = "20-day Simple Moving Average for HSBA.L stock"
label_txt = "HSBA.L Adj Close"

st.pyplot(sma())
st.write('The SMA follows the time series removing noise from the signal and keeping the relevant information about the trend. If the stock price is above its moving average it is assumed that it will likely continue rising in an uptrend.')

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.subheader('Moving Average Crossover Strategy')
st.write('The most popular moving average crossover strategy, and the "Hello World!" of quantitative trading, being the easiest to construct, is based on the simple moving average. When moving averages cross, it is usually confirmation of a change in the prevailing trend, and we want to test whether over the long-term the lag caused by the moving average can still give us profitable trades.')

st.write('Depending on the type of investor or trader (high risk vs. low risk, short-term vs. long-term trading), you can adjust your moving ‘time’ average (10 days, 20 days, 50 days, 200 days, 1 year, 5 years, etc). The longer the period of an SMA, the longer the time horizon of the trend it spots. The most commonly used SMA periods are 20 for short-term (swing) trading, 50 for medium-term (position) trading, and 200 for long-term (portfolio) trading.')

st.write('There is no single right answer and this will vary according to whether a trader is planning to buy when the trend is going down and sell when it is going up, potentially making short-term gains, or to hold for more long-term investment.')

def sma2():
  plt.figure(figsize=(15,9))
  ftse100_stocks[ticker]['Adj Close'].loc['2020-01-01':'2023-12-31'].rolling(window=20).mean().plot(label='20 Day Avg')
  ftse100_stocks[ticker]['Adj Close'].loc['2020-01-01':'2023-12-31'].rolling(window=50).mean().plot(label='50 Day Avg')
  ftse100_stocks[ticker]['Adj Close'].loc['2020-01-01':'2023-12-31'].rolling(window=200).mean().plot(label='200 Day Avg')
  ftse100_stocks[ticker]['Adj Close'].loc['2020-01-01':'2023-12-31'].plot(label=f"{label_txt}")
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15);
  plt.legend()

ticker = 'HSBA.L'

title_txt = "20, 50 and 200 day moving averages for HSBA.L stock"
label_txt = "HSBA.L Adj Close"

st.pyplot(sma2())


st.write('The chart shows that the 20-day moving average is the most sensitive to local changes, and the 200-day moving average the least. Here, the 200-day moving average indicates an overall bullish trend - the stock is trending upward over time. The 20- and 50-day moving averages are at times bearish and at other times bullish.')

st.write('The major drawback of moving averages, however, is that because they are lagging, and smooth out prices, they tend to recognize reversals too late and are therefore not very helpful when used alone.')

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.write('For statistical accuracy, we should plot the same 20, 50 and 200 days MA for a company of the same sector, however, we happened to select HSBC - the only company from banking in the list of our honorable guinea pigs. So let us do a GlaxoSmithKline Adjusted Close price data for the same time period, just for the sake of it.') 

def sma3():
  plt.figure(figsize=(15,9))
  ftse100_stocks[ticker]['Adj Close'].loc['2020-01-01':'2023-12-31'].rolling(window=20).mean().plot(label='20 Day Avg')
  ftse100_stocks[ticker]['Adj Close'].loc['2020-01-01':'2023-12-31'].rolling(window=50).mean().plot(label='50 Day Avg')
  ftse100_stocks[ticker]['Adj Close'].loc['2020-01-01':'2023-12-31'].rolling(window=200).mean().plot(label='200 Day Avg')
  ftse100_stocks[ticker]['Adj Close'].loc['2020-01-01':'2023-12-31'].plot(label=f"{label_txt}")
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15);
  plt.legend()

ticker = 'GSK.L'

title_txt = "20, 50 and 200 day moving averages for GSK.L stock"
label_txt = "GSK.L Adj Close"

st.pyplot(sma2())

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.subheader('Trading Strategy - Moving Average Crossover')
st.write('The moving average crossover trading strategy will be to take two moving averages - 20-day (fast) and 200-day (slow) - and to go long (buy) when the fast MA goes above the slow MA and to go short (sell) when the fast MA goes below the slow MA.')

st.write('Create copy of dataframe for HSBC data for 2014-2024 to work with further')

hsba_sma = hsba.copy()
st.dataframe(hsba_sma)

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.write('Let us calculate and add columns for moving averages of Adjusted Close price data')
  
hsba_sma["20d"] = np.round(hsba_sma["Adj Close"].rolling(window = 20, center = False).mean(), 2)
hsba_sma["50d"] = np.round(hsba_sma["Adj Close"].rolling(window = 50, center = False).mean(), 2)
hsba_sma["200d"] = np.round(hsba_sma["Adj Close"].rolling(window = 200, center = False).mean(), 2)

st.dataframe(hsba_sma.tail())

txt = "20, 50 and 200 day moving averages for HSBA.L stock"

st.write('Slice rows to plot data from 2019-2023')
st.pyplot(pandas_candlestick_ohlc(hsba_sma.loc['2019-01-01':'2023-12-31',:], otherseries = ["20d", "50d", "200d"]))

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Backtesting')
st.write('Before using the strategy we will evaluate its efficiency first by backtesting, or looking at how profitable it is on historical data.')
st.write('First let us identify when the 20-day average is below the 200-day average, and vice versa.')

hsba_sma['20d-200d'] = hsba_sma['20d'] - hsba_sma['200d']
st.dataframe(hsba_sma.tail())
#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.write('The sign of this difference is the regime; that is, if the fast moving average is above the slow-moving average, this is a bullish regime, and a bearish regime holds when the fast-moving average is below the slow-moving average')

st.write('np.where() is a vectorized if-else function, where a condition is checked for each component of a vector, and the first argument passed is used when the condition holds, and the other passed if it does not')
st.write('We shall assume 1s for bullish regimes and 0s for everything else. Replace the values of the bearish regime with -1, and to maintain the rest of the vector, the second argument is hsba_sma["Regime"]')
hsba_sma["Regime"] = np.where(hsba_sma['20d-200d'] > 0, 1, 0)

hsba_sma["Regime"] = np.where(hsba_sma['20d-200d'] < 0, -1, hsba_sma["Regime"])
hsba_sma.loc['2019-01-01':'2023-12-31',"Regime"].plot(ylim = (-2,2)).axhline(y = 0, color = "black", lw = 2);
plt.title("Regime for HSBA.L 20- and 200-day Moving Average Crossover Strategy for 2019-2023", color = 'black', fontsize = 20)
plt.xlabel('Date', color = 'black', fontsize = 15)
plt.ylabel('Regime', color = 'black', fontsize = 15);

st.pyplot(plt)

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.write('Now le us plot the same but for a longer period: from 2014 to the end of 2023')

hsba_sma["Regime"].plot(ylim = (-2,2)).axhline(y = 0, color = "black", lw = 2);
plt.title("Regime for HSBA.L 20- and 200-day Moving Average Crossover Strategy for 2014-2023", color = 'black', fontsize = 20)
plt.xlabel('Date', color = 'black', fontsize = 15)
plt.ylabel('Regime', color = 'black', fontsize = 15);

st.pyplot(plt)

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.write('Now let us calculate the number of bullish and bearish days for the period of 2019-2023')

st.dataframe(hsba_sma["Regime"].value_counts())

st.write('So as we can see the market was bullish for 1715 days and for 604 days it was bearish. It was also neutral for 199 days for the time-period 2019-2023')
#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.write('So he is the dataframe with added Regime values')

st.dataframe(hsba_sma)
#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.write('Now let us attempt to obtain signals with -1 indicating “sell”, 1 indicating “buy”, and 0 no action based on the regime values in the dataframe. To ensure that all trades close out, temporarily change the regime of the last row to 0')
regime_orig = hsba_sma.iloc[-1, 10]
hsba_sma.iloc[-1, 10] = 0
hsba_sma["Signal"] = np.sign(hsba_sma["Regime"] - hsba_sma["Regime"].shift(1))
# Restore original regime data
hsba_sma.iloc[-1, 10] = regime_orig
hsba_sma.tail()
st.dataframe(hsba_sma.tail())

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.write('Let us plt the results')

hsba_sma["Signal"].plot(ylim = (-2, 2));
plt.title("Trading signals for HSBA.L 20- and 200-day Moving Average Crossover Strategy for 2014-2023", color = 'black', fontsize = 20)
plt.xlabel('Date', color = 'black', fontsize = 15)
plt.ylabel('Trading signal', color = 'black', fontsize = 15);

st.pyplot(plt)

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.write('Now let us disply unique counts of trading signals')

st.dataframe(hsba_sma["Signal"].value_counts())

st.write('Essentially the means that e would buy HSBC stock 9 times and sell 9 times. If we only go long 9 trades will be engaged in over the 10-year period, while if we pivot from a long to a short position every time a long position is terminated, we would engage in 14 trades total. It is worth bearing in mind that trading more frequently isn’t necessarily good as trades are never free and broker commissions are to be paid.')

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.write('Now let us try and dentify what was the price of the stock at every buy.')

st.dataframe(hsba_sma.loc[hsba_sma["Signal"] == 1, "Close"])

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.write('Now let us do the same for the price of the stock at every sell.')

st.dataframe(hsba_sma.loc[hsba_sma["Signal"] == -1, "Close"])

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.write('Let us create a dataframe with trades, including the price at the trade and the regime under which the trade is made.')

hsba_signals = pd.concat([
        pd.DataFrame({"Price": hsba_sma.loc[hsba_sma["Signal"] == 1, "Adj Close"],
                     "Regime": hsba_sma.loc[hsba_sma["Signal"] == 1, "Regime"],
                     "Signal": "Buy"}),
        pd.DataFrame({"Price": hsba_sma.loc[hsba_sma["Signal"] == -1, "Adj Close"],
                     "Regime": hsba_sma.loc[hsba_sma["Signal"] == -1, "Regime"],
                     "Signal": "Sell"}),
    ])
hsba_signals.sort_index(inplace = True)
st.dataframe(hsba_signals)

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.write('Let us see if long trades had been profitable')

hsba_long_profits = pd.DataFrame({
        "Price": hsba_signals.loc[(hsba_signals["Signal"] == "Buy") &
                                  hsba_signals["Regime"] == 1, "Price"],
        "Profit": pd.Series(hsba_signals["Price"] - hsba_signals["Price"].shift(1)).loc[
            hsba_signals.loc[(hsba_signals["Signal"].shift(1) == "Buy") & (hsba_signals["Regime"].shift(1) == 1)].index
        ].tolist(),
        "End Date": hsba_signals["Price"].loc[
            hsba_signals.loc[(hsba_signals["Signal"].shift(1) == "Buy") & (hsba_signals["Regime"].shift(1) == 1)].index
        ].index
    })
st.dataframe(hsba_long_profits)

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.subheader('Exponential Moving Average')
st.write('In a Simple Moving Average, each value in the time period carries equal weight, and values outside of the time period are not included in the average. However, the Exponential Moving Average is a cumulative calculation where a different decreasing weight is assigned to each observation. Past values have a diminishing contribution to the average, while more recent values have a greater contribution. This method allows the moving average to be more responsive to changes in the data.')
st.write('Let us establish a 20-day EMA for Adjusted Close price for year 2023')

def ewma():
  plt.figure(figsize=(15,9))
  ftse100_stocks[ticker]['Adj Close'].loc['2023-01-01':'2023-12-31'].ewm(20).mean().plot(label='20 Day Avg')
  ftse100_stocks[ticker]['Adj Close'].loc['2023-01-01':'2023-12-31'].plot(label=f"{label_txt}")
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15);
  plt.legend()

ticker = 'HSBA.L'
title_txt = "20-day Exponential Moving Average for HSBA.L stock"
label_txt = "HSBA.L Adj Close"

st.pyplot(ewma())

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 
