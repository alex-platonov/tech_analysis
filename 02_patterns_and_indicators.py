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

st.write('Let us plot the results')

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
st.write('In a Simple Moving Average (SMA), each value in the time period carries equal weight, and values outside of the time period are not included in the average. However, the Exponential Moving Average (EMA) is a cumulative calculation where a different decreasing weight is assigned to each observation. Past values have a diminishing contribution to the average, while more recent values have a greater contribution. This method allows the moving average to be more responsive to changes in the data.')
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

st.write('Let us establish 20-, 50- and 200-day EMA for Adjusted Close price for 2019-2023')

def ewma2():
  plt.figure(figsize=(15,9))
  ftse100_stocks[ticker]['Adj Close'].loc['2019-01-01':'2023-12-31'].ewm(20).mean().plot(label='20 Day Avg')
  ftse100_stocks[ticker]['Adj Close'].loc['2019-01-01':'2023-12-31'].ewm(50).mean().plot(label='50 Day Avg')
  ftse100_stocks[ticker]['Adj Close'].loc['2019-01-01':'2023-12-31'].ewm(200).mean().plot(label='200 Day Avg')
  ftse100_stocks[ticker]['Adj Close'].loc['2019-01-01':'2023-12-31'].plot(label=f"{label_txt}")
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15);
  plt.legend()

ticker = 'HSBA.L'
title_txt = "20, 50 and 200-day Exponential Moving Averages for HSBA.L stock"
label_txt = "HSBA.L Adj Close"

st.pyplot(ewma2())

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.subheader('Triple Moving Average Crossover Strategy')
st.write('This strategy uses three moving moving averages - short/fast, middle/medium and long/slow - and has two buy and sell signals.')

st.write('The first is to buy when the middle/medium moving average crosses above the long/slow moving average and the short/fast moving average crosses above the middle/medium moving average. If we use this buy signal the strategy is to sell if the short/fast moving average crosses below the middle/medium moving average.')

st.write('The second is to buy when the middle/medium moving average crosses below the long/slow moving average and the short/fast moving average crosses below the middle/medium moving average. If we use this buy signal the strategy is to sell if the short/fast moving average crosses above the middle/medium moving average.')


st.dataframe(hsba[['Adj Close']]['2023-05-01':'2023-10-31'])

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

def adj_6mo():
  sns.set(rc={'figure.figsize':(15, 9)})
  ftse100_stocks[ticker]['Adj Close'].loc['2023-05-01':'2023-10-31'].plot(label=f"{label_txt}")
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15);
  plt.legend()

ticker = 'HSBA.L'
title_txt = "HSBA.L Adjusted Close Price from 1 May - 31 Oct 2023"
label_txt = "HSBA.L Adj Close "

st.pyplot(adj_6mo())

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.write('Let us calculate three ranges of EMA: long, middle and short).')
hsba_adj_6mo = hsba[['Adj Close']]['2023-05-01':'2023-10-31']

# Calculate Short-, middle- and long EMA
ShortEMA = hsba_adj_6mo['Adj Close'].ewm(span=5, adjust=False).mean()
MiddleEMA = hsba_adj_6mo['Adj Close'].ewm(span=21, adjust=False).mean()
LongEMA = hsba_adj_6mo['Adj Close'].ewm(span=63, adjust=False).mean()

def ewma3():
  sns.set(rc={'figure.figsize':(15, 9)})
  plt.plot(ftse100_stocks[ticker]['Adj Close'].loc['2023-05-01':'2023-10-31'], label=f"{label_txt}", color = 'blue')
  plt.plot(ShortEMA, label = 'Short/Fast EMA', color = 'red')
  plt.plot(MiddleEMA, label = 'Middle/Medium EMA', color = 'orange')
  plt.plot(LongEMA, label = 'Long/Slow EMA', color = 'green')
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15);
  plt.legend()

ticker = 'HSBA.L'
title_txt = "Triple Exponential Moving Average Crossover for HSBA.L stock"
label_txt = "HSBA.L Adj Close"

st.pyplot(ewma3())

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.write('Now let us see how they look in data:')

hsba_adj_6mo['Short'] = ShortEMA
hsba_adj_6mo['Middle'] = MiddleEMA
hsba_adj_6mo['Long'] = LongEMA

st.write('Short EMA')
st.dataframe(hsba_adj_6mo['Short'])

st.write('Middle EMA')
st.dataframe(hsba_adj_6mo['Middle'])

st.write('Long EMA')
st.dataframe(hsba_adj_6mo['Long'])

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.write('Let us attempt to define BUY and SELL signals')

def buy_sell_ewma3(data):
  
  buy_list = []
  sell_list = []
  flag_long = False
  flag_short = False

  for i in range(0, len(data)):
    if data['Middle'][i] < data['Long'][i] and data['Short'][i] < data['Middle'][i] and flag_long == False and flag_short == False:
      buy_list.append(data['Adj Close'][i])
      sell_list.append(np.nan)
      flag_short = True
    elif flag_short == True and data['Short'][i] > data['Middle'][i]:
      sell_list.append(data['Adj Close'][i])
      buy_list.append(np.nan)
      flag_short = False
    elif data['Middle'][i] > data['Long'][i] and data['Short'][i] > data['Middle'][i] and flag_long == False and flag_short == False:
      buy_list.append(data['Adj Close'][i])
      sell_list.append(np.nan)
      flag_long = True
    elif flag_long == True and data['Short'][i] < data['Middle'][i]:
      sell_list.append(data['Adj Close'][i])
      buy_list.append(np.nan)
      flag_long = False
    else:
      buy_list.append(np.nan)
      sell_list.append(np.nan)
  
  return (buy_list, sell_list)

hsba_adj_6mo['Buy'] = buy_sell_ewma3(hsba_adj_6mo)[0]
hsba_adj_6mo['Sell'] = buy_sell_ewma3(hsba_adj_6mo)[1]

def buy_sell_ewma3_plot():
  sns.set(rc={'figure.figsize':(18, 10)})
  plt.plot(ftse100_stocks[ticker]['Adj Close'].loc['2023-05-01':'2023-10-31'], label=f"{label_txt}", color = 'blue', alpha = 0.35)
  plt.plot(ShortEMA, label = 'Short/Fast EMA', color = 'red', alpha = 0.35)
  plt.plot(MiddleEMA, label = 'Middle/Medium EMA', color = 'orange', alpha = 0.35)
  plt.plot(LongEMA, label = 'Long/Slow EMA', color = 'green', alpha = 0.35)
  plt.scatter(hsba_adj_6mo.index, hsba_adj_6mo['Buy'], color = 'green', label = 'Buy Signal', marker = '^', alpha = 1)
  plt.scatter(hsba_adj_6mo.index, hsba_adj_6mo['Sell'], color = 'red', label = 'Buy Signal', marker='v', alpha = 1)
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15);
  plt.legend()

ticker = 'HSBA.L'
title_txt = "Trading signals for HSBA.L stock"
label_txt = "HSBA.L Adj Close"

st.pyplot(buy_sell_ewma3_plot())

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 

st.subheader('Exponential Smoothing')
st.write('Single Exponential Smoothing, also known as Simple Exponential Smoothing, is a time series forecasting method for univariate data without a trend or seasonality. It requires an alpha parameter, also called the smoothing factor or smoothing coefficient, to control the rate at which the influence of the observations at prior time steps decay exponentially.')

def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def plot_exponential_smoothing(series, alphas):
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        plt.plot(exponential_smoothing(series, alpha), label=f"Alpha {alpha}")
    plt.plot(series.values, "c", label = f"{label_txt}")
    plt.xlabel('Days', color = 'black', fontsize = 15)
    plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15);
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title(f"{title_txt}", color = 'black', fontsize = 20)
    plt.grid(True);

ticker = 'HSBA.L'
title_txt = "Single Exponential Smoothing for HSBA.L stock using 0.05 and 0.3 as alpha values"
label_txt = "HSBA.L Adj Close"

st.pyplot(plot_exponential_smoothing(ftse100_stocks[ticker]['Adj Close'].loc['2019-01-01':'2023-12-31'], [0.05, 0.3]))

st.write('The smaller the smoothing factor (coefficient), the smoother the time series will be. As the smoothing factor approaches 0, we approach the moving average model so the smoothing factor of 0.05 produces a smoother time series than 0.3. This indicates slow learning (past observations have a large influence on forecasts). A value close to 1 indicates fast learning (that is, only the most recent values influence the forecasts). Double Exponential Smoothing (Holt’s Linear Trend Model) is an extension being a recursive use of Exponential Smoothing twice where beta is the trend smoothing factor, and takes values between 0 and 1. It explicitly adds support for trends.')

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True) 


def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result

def plot_double_exponential_smoothing(series, alphas, betas):
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        for beta in betas:
            plt.plot(double_exponential_smoothing(series, alpha, beta), label=f"Alpha {alpha}, beta {beta}")
    plt.plot(series.values, label = f"{label_txt}")
    plt.xlabel('Days', color = 'black', fontsize = 15)
    plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15)
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title(f"{title_txt}", color = 'black', fontsize = 20)
    plt.grid(True)

ticker = 'HSBA.L'
title_txt = "Double Exponential Smoothing for HSBA.L stock with different alpha and beta values"
label_txt = "HSBA.L Adj Close"

st.pyplot(plot_double_exponential_smoothing(ftse100_stocks[ticker]['Adj Close'].loc['2019-01-01':'2023-12-31'], alphas=[0.9, 0.02],))
          
st.write('The third main type is Triple Exponential Smoothing aka Holt Winters Method, which is an extension of Exponential Smoothing that explicitly adds support for seasonality or periodic fluctuations. Since we are analyzing a bank sector stock we shall omit the triple smoothing as seasonality and any periodic fluctuations do not have a drastic effect on the whole picture.')
#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)          

st.subheader('Moving average convergence divergence (MACD)')
st.write('The MACD is a trend-following momentum indicator turning two trend-following indicators, moving averages, into a momentum oscillator by subtracting the longer moving average from the shorter one.')

st.write('It is useful although lacking one prediction element - because it is unbounded it is not particularly useful for identifying overbought and oversold levels. Traders can look for signal line crossovers, neutral/centreline crossovers (otherwise known as the 50 level) and divergences from the price action to generate signals.')

st.write('The default parameters are 26 EMA of prices, 12 EMA of prices and a 9-moving average of the difference between the first two.')

def adj_3mo():
  sns.set(rc={'figure.figsize':(15, 9)})
  ftse100_stocks[ticker]['Adj Close'].loc['2023-08-01':'2023-10-31'].plot(label=f"{label_txt}")
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15);
  plt.legend()

ticker = 'HSBA.L'
title_txt = "HSBA.L Adjusted Close Price from 1 Aug - 31 Oct 2023"
label_txt = "HSBA.L Adj Close "

st.write('Let us plot HSBC Adjusted close prices for 3 months in 2023 just for reference purposes')
st.pyplot(adj_3mo())

hsba_adj_3mo = hsba[['Adj Close']]['2023-08-01':'2023-10-31']

st.write('In order to correctly calculate the MACD we should take the following values: Short EMA (as defined before), Long EMA (as defined before), their difference, and the resulting signal.')

ShortEMA = hsba_adj_3mo['Adj Close'].ewm(span=12, adjust=False).mean()
LongEMA = hsba_adj_3mo['Adj Close'].ewm(span=26, adjust=False).mean()
MACD = ShortEMA - LongEMA
signal = MACD.ewm(span=9, adjust=False).mean()

def macd():
  plt.figure(figsize=(15, 9))
  plt.plot(hsba_adj_3mo.index, MACD, label = f"{macd_label_txt}", color= 'red')
  plt.plot(hsba_adj_3mo.index, signal, label = f"{sig_label_txt}", color= 'blue')
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xticks (rotation = 45)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.legend(loc='upper left')
  plt.show()

ticker = 'HSBA.L'
title_txt = 'MACD and Signal line for HSBA.L stock from 1 Aug - 31 Oct 2023'
macd_label_txt ="HSBA.L MACD"
sig_label_txt = "Signal Line"

st.pyplot(macd())

st.write('When the MACD line crosses above the signal line this indicates a good time to buy.')

st.write('Now let us try to display the signals fr BUY and SELL')

# Function to signal when to buy and sell

def buy_sell_macd(signal):
  Buy = []
  Sell = []
  flag = -1

  for i in range(0, len(signal)):
    if signal['MACD'][i] > signal['Signal Line'][i]:
      Sell.append(np.nan)
      if flag != 1:
        Buy.append(signal['Adj Close'][i])
        flag = 1
      else:
        Buy.append(np.nan)
    elif signal['MACD'][i] < signal['Signal Line'][i]:
      Buy.append(np.nan)
      if flag != 0:
        Sell.append(signal['Adj Close'][i])
        flag = 0
      else:
        Sell.append(np.nan)
    else:
      Buy.append(np.nan)
      Sell.append(np.nan)

  return (Buy, Sell)

# Create buy and sell columns

a = buy_sell_macd(hsba_adj_3mo)
hsba_adj_3mo['Buy_Signal_Price'] = a[0]
hsba_adj_3mo['Sell_Signal_Price'] = a[1]

# Plot buy and sell signals

def buy_sell_macd_plot():
  plt.figure(figsize=(20, 10))
  plt.scatter(hsba_adj_3mo.index, hsba_adj_3mo['Buy_Signal_Price'], color='green', label='Buy', marker='^', alpha=1)
  plt.scatter(hsba_adj_3mo.index, hsba_adj_3mo['Sell_Signal_Price'], color='red', label='Sell', marker='v', alpha=1)
  plt.plot(hsba_adj_3mo['Adj Close'], label='Adj Close Price', alpha = 0.35)
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Adj Close Price')
  plt.legend(loc = 'upper left')
  plt.show()

ticker = 'HSBA.L'
title_txt = 'HSBA.L Adjusted Close Price Buy & Sell Signals'

st.pyplot(buy_sell_macd_plot())

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)   

st.header('Momentum Strategies')
st.write('In momentum algorithmic trading strategies stocks have momentum (i.e. upward or downward trends) that we can detect and exploit.') 

st.subheader('Relative Strength Index (RSI)') 
st.write('The RSI is a momentum indicator. A typical momentum strategy will buy stocks that have been showing an upward trend in hopes that the trend will continue, and make predictions based on whether the past recent values were going up or going down.')

st.write('The RSI determines the level of overbought (70) and oversold (30) zones using a default lookback period of 14 i.e. it uses the last 14 values to calculate its values. The idea is to buy when the RSI touches the 30 barrier and sell when it touches the 70 barrier.')

def adj_12mo():
  sns.set(rc={'figure.figsize':(15, 9)})
  ftse100_stocks[ticker]['Adj Close'].loc['2023-01-01':'2023-12-31'].plot(label=f"{label_txt}")
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15);
  plt.legend()

st.write('Let us again display adjusted close prices graph for 2023 for reference') 

ticker = 'HSBA.L'
title_txt = "HSBA.L Adjusted Close Price from 1 Jan - 31 Dec 2023"
label_txt = "HSBA.L Adj Close "

st.pyplot(adj_12mo())
#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)   

hsba_adj_12mo = hsba[['Adj Close']]['2023-01-01':'2023-12-31']

st.write('In order to be able to calculate the RSI,  first our data needs to be prepared.')

st.write('First, let us get difference in price for the previous day.')

delta = hsba_adj_12mo['Adj Close'].diff(1)
st.dataframe(delta)

# Remove NaNs

delta = delta.dropna()

st.write('Second, let us get positive gains (up) and negative gains (down)')
up = delta.copy()
down = delta.copy()
up[up < 0] = 0
down[down > 0] = 0
st.dataframe(up)
st.dataframe(down)

st.write('Third, let us get the time period:')
period = 14
st.dataframe(period)


st.write('Fourth, let us calculate average gain and average loss')
AVG_Gain = up.rolling(window=period).mean()
st.dataframe(AVG_Gain)

#AVG_Loss = abs(down.rolling(window=period).mean())
AVG_Loss = down.abs().rolling(window=period).mean()
st.dataframe(AVG_Loss)

st.write('And now let us calculate RSI based on SMA.')

# Calculate Relative Strength (RS)
RS = AVG_Gain / AVG_Loss
# Calculate RSI
RSI = 100.0 - (100.0 / (1.0 + RS))
def rsi():
  sns.set(rc={'figure.figsize':(20, 10)})
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('RSI', color = 'black', fontsize = 15);
  RSI.plot()
ticker = 'HSBA.L'
title_txt = "HSBA.L RSI plot for 1 Jan - 31 Dec 2023"
label_txt = "HSBA.L RSI level"

st.pyplot(rsi())

st.write('Now, let us create a dataframe with Adjusted Close and RSI together.')
new_df = pd.DataFrame()
new_df['Adj Close'] = hsba_adj_12mo['Adj Close']
new_df['RSI'] = RSI
st.dataframe(new_df)

st.write('And now, let us plot HSBC adjusted close price for the whole 2023 year.')
def adj_close_12mo():
  sns.set(rc={'figure.figsize':(20, 10)})
  plt.plot(new_df.index, new_df['Adj Close'])
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Stock Price (p)', color = 'black', fontsize = 15);
  plt.legend(new_df.columns.values, loc = 'upper left')
  plt.show()

title_txt = "HSBA.L Adjusted Close Price from 1 Jan - 31 Dec 2023"

st.pyplot(adj_close_12mo())

st.write('Let us plot the corresponding RSI values and the significant levels that we had calculated earlier.')

def rsi_sma():
  plt.figure(figsize=(20, 10))
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.plot(new_df.index, new_df['RSI'])
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.axhline(0, linestyle='--', alpha = 0.5, color='gray')
  plt.axhline(10, linestyle='--', alpha = 0.5, color='orange')
  plt.axhline(20, linestyle='--', alpha = 0.5, color='green')
  plt.axhline(30, linestyle='--', alpha = 0.5, color='red')
  plt.axhline(70, linestyle='--', alpha = 0.5, color='red')
  plt.axhline(80, linestyle='--', alpha = 0.5, color='green')
  plt.axhline(90, linestyle='--', alpha = 0.5, color='orange')
  plt.axhline(100, linestyle='--', alpha = 0.5, color='gray')
  plt.show()
title_txt = 'HSBA.L RSI based on SMA'

st.pyplot(rsi_sma())


st.write('Now let us us attempt the same excersise but based on EWMA.')
period = 14

# Calculate the EWMA average gain and average loss
AVG_Gain2 = up.ewm(span=period).mean()
AVG_Loss2 = down.abs().ewm(span=period).mean()

# Calculate the RSI based on EWMA
RS2 = AVG_Gain2 / AVG_Loss2
RSI2 = 100.0 - (100.0 / (1.0 + RS2))

new_df2 = pd.DataFrame()
new_df2['Adj Close'] = hsba_adj_12mo['Adj Close']
new_df2['RSI2'] = RSI2
st.dataframe(new_df2)

# Plot corresponding RSI values and the significant levels

def rsi_ewma():
  plt.figure(figsize=(20, 10))
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.plot(new_df2.index, new_df2['RSI2'])
  plt.axhline(0, linestyle='--', alpha = 0.5, color='gray')
  plt.axhline(10, linestyle='--', alpha = 0.5, color='orange')
  plt.axhline(20, linestyle='--', alpha = 0.5, color='green')
  plt.axhline(30, linestyle='--', alpha = 0.5, color='red')
  plt.axhline(70, linestyle='--', alpha = 0.5, color='red')
  plt.axhline(80, linestyle='--', alpha = 0.5, color='green')
  plt.axhline(90, linestyle='--', alpha = 0.5, color='orange')
  plt.axhline(100, linestyle='--', alpha = 0.5, color='gray')
  plt.show()
title_txt = 'HSBA.L RSI based on EWMA'

st.pyplot(rsi_ewma())

st.write('A conclusion of sorts: it appears that RSI value dips below the 20 significant level in November 2023 indicating that the stock was oversold and presented a buying opportunity for an investor before a price rise.')

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)   

st.subheader('Money Flow Index (MFI)')
st.write('Money Flow Index (MFI) is a technical oscillator, and momentum indicator, that uses price and volume data for identifying overbought or oversold signals in an asset. It can also be used to spot divergences that warn of a trend change in price. The oscillator moves between 0 and 100 and a reading of above 80 implies overbought conditions, and below 20 implies oversold conditions.')

st.write('It is related to the Relative Strength Index (RSI) but incorporates volume, whereas the RSI only considers price.')
st.write(' Let us again experiment on HSBC data for 2023')

hsba_12mo = hsba.copy()  
hsba_12mo = hsba_12mo['2023-01-01':'2023-12-31']
st.dataframe(hsba_12mo)

def hsba_12mo_close():
  plt.figure(figsize=(20, 10))
  plt.plot(hsba_12mo['Close'])
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Close Price', color = 'black', fontsize = 15)
  plt.legend(hsba_12mo.columns.values, loc='upper left')
  plt.show()

title_txt = "HSBA.L Close Price from 1 Jan - 31 Dec 2023"
label_txt = "HSBA.L Close price "

st.pyplot(hsba_12mo_close())


st.write('Let us calculate a typical price...')

typical_price = (hsba_12mo['Close'] + hsba_12mo['High'] + hsba_12mo['Low']) / 3
st.dataframe(typical_price)

period = 14
st.write('and the money flow.')
money_flow = typical_price * hsba_12mo['Volume']
st.dataframe(money_flow)

# Get all positive and negative money flows
positive_flow = []
negative_flow = []

# Loop through typical price
for i in range(1, len(typical_price)):
  if typical_price[i] > typical_price[i-1]:
    positive_flow.append(money_flow[i-1])
    negative_flow.append(0)
  elif typical_price[i] < typical_price[i-1]:
    negative_flow.append(money_flow[i-1])
    positive_flow.append(0)
  else:
    positive_flow.append(0)
    negative_flow.append(0)
    
# Get all positive and negative money flows within same time period
positive_mf = []
negative_mf = []

for i in range(period-1, len(positive_flow)):
  positive_mf.append(sum(positive_flow[i + 1 - period : i+1]))
for i in range(period-1, len(negative_flow)):
  negative_mf.append(sum(negative_flow[i + 1 - period : i+1]))

st.write('Now let us calculate money flow index:')

mfi = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf)))
st.dataframe(mfi)

df2 = pd.DataFrame()
df2['MFI'] = mfi
st.write(' And create the plot:')

def mfi_plot():
  plt.figure(figsize=(20, 10))
  plt.plot(df2['MFI'], label = 'MFI')
  plt.axhline(10, linestyle = '--', color = 'orange')
  plt.axhline(20, linestyle = '--', color = 'blue')
  plt.axhline(80, linestyle = '--', color = 'blue')
  plt.axhline(90, linestyle = '--', color = 'orange')
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Time periods', color = 'black', fontsize = 15)
  plt.ylabel('MFI Values', color = 'black', fontsize = 15)
  plt.show()
title_txt = "HSBA.L MFI"

st.pyplot(mfi_plot())

st.write('Let us now attempt and create definite BUY and SELL signals based on this information. Basically,  data-wise the process is the same as for previous strategies: we create a copy of the dataframe, create a function to calculate BUY and SELL signals, add those new columns to the dataframe, and then try to plot the result')

new_mfi_df = pd.DataFrame()
new_mfi_df = hsba_12mo[period:]
new_mfi_df['MFI'] = mfi

# Create function to get buy and sell signals

def get_signal(data, high, low):
  buy_signal = []
  sell_signal = []

  for i in range(len(data['MFI'])):
    if data['MFI'][i] > high:
      buy_signal.append(np.nan)
      sell_signal.append(data['Close'][i])
    elif data['MFI'][i] < low:
      buy_signal.append(data['Close'][i])
      sell_signal.append(np.nan)
    else:
      sell_signal.append(np.nan)
      buy_signal.append(np.nan)

  return (buy_signal, sell_signal)
# Add new columns (Buy & Sell)

new_mfi_df['Buy'] = get_signal(new_mfi_df, 80, 20)[0]
new_mfi_df['Sell'] = get_signal(new_mfi_df, 80, 20)[1]

def mfi_buy_sell_plot():
  plt.figure(figsize=(20, 10))
  plt.plot(new_mfi_df['Close'], label = 'Close Price', alpha = 0.5)
  plt.scatter(new_mfi_df.index, new_mfi_df['Buy'], color = 'green', label = 'Buy Signal', marker = '^', alpha = 1)
  plt.scatter(new_mfi_df.index, new_mfi_df['Sell'], color = 'red', label = 'Sell Signal', marker = 'v', alpha = 1)
  plt.title(f"{title_txt}", color = 'black', fontsize = 20)
  plt.xlabel('Date', color = 'black', fontsize = 15)
  plt.ylabel('Close Price', color = 'black', fontsize = 15)
  plt.legend(loc='upper left')
  plt.show()
title_txt = "Trading signals for HSBA.L stock"

st.pyplot(mfi_buy_sell_plot())

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)   

st.subheader('Stochastic Oscillator')
st.write('The stochastic oscillator is a momentum indicator comparing the closing price of a security to the range of its prices over a certain period of time and is one of the best-known momentum indicators along with RSI and MACD.')
st.write('The intuition is that in a market trending upward, prices will close near the high, and in a market trending downward, prices close near the low.')
st.write('The stochastic oscillator is plotted within a range of zero and 100. The default parameters are an overbought zone of 80, an oversold zone of 20 and well-used lookbacks period of 14 and 5 which can be used simultaneously. The oscillator has two lines, the %K and %D, where the former measures momentum and the latter measures the moving average of the former. The %D line is more important of the two indicators and tends to produce better trading signals which are created when the %K crosses through the %D.')

hsba_so = hsba.copy()

st.markdown("""
Stochastic Oscillator Formula

The stochastic oscillator is calculated using the following formula:

`%K = 100(C – L14) / (H14 – L14)`

Where:

- `C` = the most recent closing price
- `L14` = the low of the 14 previous trading sessions
- `H14` = the highest price traded during the same 14-day period
- `%K` = the current market rate for the currency pair
- `%D` = 3-period moving average of `%K`

The formula helps determine the momentum of a stock or asset by comparing the closing price to the high and low prices over a certain period of time, typically 14 days.
""", unsafe_allow_html=True)

st.write('In this implementation there are 3 possible states – long, short, flat (i.e. no position).')
st.write('For correct calculations L14, H14, %K and %D columns need to be created and appended to the dataframe')

#Create the "L14" column in the DataFrame
hsba_so['L14'] = hsba_so['Low'].rolling(window=14).min()

#Create the "H14" column in the DataFrame
hsba_so['H14'] = hsba_so['High'].rolling(window=14).max()

#Create the "%K" column in the DataFrame
hsba_so['%K'] = 100*((hsba_so['Close'] - hsba_so['L14']) / (hsba_so['H14'] - hsba_so['L14']))

#Create the "%D" column in the DataFrame
hsba_so['%D'] = hsba_so['%K'].rolling(window=3).mean()


st.write('Let us create a plot (with 2 subplots) showing the HSBA.L price over time, along with a visual representation of the Stochastic Oscillator.')

fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(20,10))
fig.subplots_adjust(hspace=0.5)

hsba_so['Close'].plot(ax=axes[0]); axes[0].set_title('Close')
hsba_so[['%K','%D']].plot(ax=axes[1]); axes[1].set_title('Oscillator');
st.pyplot(fig)

st.write('Before going fyther data needs to be manipulated in a certain way in order to accomodate for our further calculations.')
st.markdown("""
The following manipulations are to be performed: 
- Create a column in the DataFrame showing "TRUE" if sell entry signal is given and "FALSE" otherwise. A sell is initiated when the %K line crosses down through the %D line and the value of the oscillator is above 80;
- Create a column in the DataFrame showing "TRUE" if sell exit signal is given and "FALSE" otherwise. A sell exit signal is given when the %K line crosses back up through the %D line;
- Create a placeholder column to populate with short positions (-1 for short and 0 for flat) using boolean values created above;
- Set initial position on day 1 to flat;
- Forward fill the position column to represent the holding of positions through time;
- Create a column in the DataFrame showing "TRUE" if buy entry signal is given and "FALSE" otherwise. A buy is initiated when the %K line crosses up through the %D line and the value of the oscillator is below 20;
- Create a column in the DataFrame showing "TRUE" if buy exit signal is given and "FALSE" otherwise. A buy exit signal is given when the %K line crosses back down through the %D line;
- Create a placeholder column to populate with long positions (1 for long and 0 for flat) using boolean values created above;
- Set initial position on day 1 to flat;
- Forward fill the position column to represent the holding of positions through time;
- Add Long and Short positions together to get final strategy position (1 for long, -1 for short and 0 for flat);
""", unsafe_allow_html=True)
#First, let us create a column in the DataFrame showing "TRUE" if sell entry signal is given and "FALSE" otherwise. A sell is initiated when the %K line crosses down through the %D line and the value of the oscillator is above 80:
hsba_so['Sell Entry'] = ((hsba_so['%K'] < hsba_so['%D']) & (hsba_so['%K'].shift(1) > hsba_so['%D'].shift(1))) & (hsba_so['%D'] > 80)
#Second, let us create a column in the DataFrame showing "TRUE" if sell exit signal is given and "FALSE" otherwise. A sell exit signal is given when the %K line crosses back up through the %D line: 
hsba_so['Sell Exit'] = ((hsba_so['%K'] > hsba_so['%D']) & (hsba_so['%K'].shift(1) < hsba_so['%D'].shift(1)))

#Third, let us create a placeholder column to populate with short positions (-1 for short and 0 for flat) using boolean values created above:
hsba_so['Short'] = np.nan 
hsba_so.loc[hsba_so['Sell Entry'],'Short'] = -1 
hsba_so.loc[hsba_so['Sell Exit'],'Short'] = 0 

#Set initial position on day 1 to flat 
hsba_so['Short'][0] = 0 

#Forward fill the position column to represent the holding of positions through time 
hsba_so['Short'] = hsba_so['Short'].fillna(method='pad') 

#Create a column in the DataFrame showing "TRUE" if buy entry signal is given and "FALSE" otherwise. 
#A buy is initiated when the %K line crosses up through the %D line and the value of the oscillator is below 20 
hsba_so['Buy Entry'] = ((hsba_so['%K'] > hsba_so['%D']) & (hsba_so['%K'].shift(1) < hsba_so['%D'].shift(1))) & (hsba_so['%D'] < 20) 

#Create a column in the DataFrame showing "TRUE" if buy exit signal is given and "FALSE" otherwise. 
#A buy exit signal is given when the %K line crosses back down through the %D line 
hsba_so['Buy Exit'] = ((hsba_so['%K'] < hsba_so['%D']) & (hsba_so['%K'].shift(1) > hsba_so['%D'].shift(1))) 

#create a placeholder column to populate with long positions (1 for long and 0 for flat) using boolean values created above 
hsba_so['Long'] = np.nan  
hsba_so.loc[hsba_so['Buy Entry'],'Long'] = 1  
hsba_so.loc[hsba_so['Buy Exit'],'Long'] = 0  

#Set initial position on day 1 to flat 
hsba_so['Long'][0] = 0  

#Forward fill the position column to represent the holding of positions through time 
hsba_so['Long'] = hsba_so['Long'].fillna(method='pad') 

#Add Long and Short positions together to get final strategy position (1 for long, -1 for short and 0 for flat) 
hsba_so['Position'] = hsba_so['Long'] + hsba_so['Short']

st.write('This data manipulation has been done under the covers, however if you wish to see the raw process, please go to https://github.com/alex-platonov/tech_analysis/blob/main/02_patterns_and_indicators.ipynb and see for yourself')
st.write('Now we can plot the position through time to get an idea of when we are long and when we are short:)

st.pyplot(hsba_so['Position'].plot(figsize=(20,10));)

st.write('Now let us attempt to plot the strategy returns versus the underlying HSBA stock returns over a period of time.')
#Set up a column holding the daily HSBA.L returns
hsba_so['Market Returns'] = hsba_so['Close'].pct_change()

#Create column for Strategy Returns by multiplying the daily HSAB.L returns by the position that was held at close
#of business the previous day
hsba_so['Strategy Returns'] = hsba_so['Market Returns'] * hsba_so['Position'].shift(1)

#Finally plot the strategy returns versus HSBA.L returns
hsba_so[['Strategy Returns','Market Returns']].cumsum().plot(figsize=(20,10));

st.pyplot(plt.title('Strategy returns versus HSBA.L returns', color = 'black', fontsize = 20);)

st.write('So here we can see that the returns were somewhat positive but by a minuscule rate. The stock graph exhibits some violent volatility so, retrospectively the strategy of buying and holding would not  have had any significant returns, however, short-term speculative approach might have resulted in significant gains or losses.')

st.write('It is worth mentioning that this strategy has a second implementation – the one where we are either holding long or short positions.')

hsba_so['L14'] = hsba_so['Low'].rolling(window=14).min()
hsba_so['H14'] = hsba_so['High'].rolling(window=14).max()

hsba_so['%K'] = 100*((hsba_so['Close'] - hsba_so['L14']) / (hsba_so['H14'] - hsba_so['L14']) )
hsba_so['%D'] = hsba_so['%K'].rolling(window=3).mean()

hsba_so['Sell Entry'] = ((hsba_so['%K'] < hsba_so['%D']) & (hsba_so['%K'].shift(1) > hsba_so['%D'].shift(1))) & (hsba_so['%D'] > 80)
hsba_so['Buy Entry'] = ((hsba_so['%K'] > hsba_so['%D']) & (hsba_so['%K'].shift(1) < hsba_so['%D'].shift(1))) & (hsba_so['%D'] < 20)

st.markdown(""" Some data manipulation is essential for further calculations: 
- An empty column 'Position needs to be created in the dataframe;
- Set position to -1 for sell signals;
- Set position to -1 for buy signals;
- Set starting position to flat (i.e. 0);
- Forward fill the position column to show holding of positions through time; 
- Set up a column holding the daily HSBA.L returns; 
- Create column for Strategy Returns by multiplying the daily HSBA.L returns by the position that was held at the close of business the previous day;

""", unsafe_allow_html=True)

#Create empty "Position" column
hsba_so['Position'] = np.nan 

#
hsba_so.loc[hsba_so['Sell Entry'],'Position'] = -1 

#Set position to -1 for buy signals
hsba_so.loc[hsba_so['Buy Entry'],'Position'] = 1 

#Set starting position to flat (i.e. 0)
hsba_so['Position'].iloc[0] = 0 

#Forward fill the position column to show holding of positions through time
hsba_so['Position'] = hsba_so['Position'].fillna(method='ffill')

#Set up a column holding the daily HSBA.L returns
hsba_so['Market Returns'] = hsba_so['Close'].pct_change()

#Create column for Strategy Returns by multiplying the daily HSBA.L returns by the position that was held at close
#of business the previous day
hsba_so['Strategy Returns'] = hsba_so['Market Returns'] * hsba_so['Position'].shift(1)

st.write('Finally we can plot the strategy returns versus HSBA.L returns')
hsba_so[['Strategy Returns','Market Returns']].cumsum().plot(figsize=(20,10));

plt.title('Strategy returns versus HSBA.L returns (long or short)', color = 'black', fontsize = 20);

st.write('This strategy implementation demonstrates better outcomes.')

#--------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)   

st.subheader('Rate of Change (ROC)')
st.write('The ROC indicator is a pure momentum oscillator. The ROC calculation compares the current price with the price "n" periods ago e.g. when we compute the ROC of the daily price with a 9-day lag, we are simply looking at how much, in percentage, the price has gone up (or down) compared to 9 days ago. Like other momentum indicators, ROC has overbought and oversold zones that may be adjusted according to market conditions.')

st.write('As before, the experiment starts with clean duplicating the existing dataframe:')
hsba_roc = hsba.copy()
st.dataframe(hsba_roc)

hsba_roc_12mo = hsba_roc['2023-01-01':'2023-12-31']
hsba_roc_12mo['ROC'] = ( hsba_roc_12mo['Adj Close'] / hsba_roc_12mo['Adj Close'].shift(9) -1 ) * 100

st.write('Let us take only last 100 days as a subset to experiment on.')

hsba_roc_100d = hsba_roc_12mo[-100:]
dates = hsba_roc_100d.index
price = hsba_roc_100d['Adj Close']
roc = hsba_roc_100d['ROC']
st.dataframe(hsba_roc_12mo[-100:])

st.write('Now let us plot HSBA.L Adj Close Price and 9-day ROC for last 100 days of 2023.')

fig = plt.figure(figsize=(16,10))
fig.subplots_adjust(hspace=0)

plt.rcParams.update({'font.size': 14})

# Price subplot
price_ax = plt.subplot(2, 1, 1)
price_ax.plot(dates, price, color='blue', linewidth=2, label="Adj Closing Price")
price_ax.legend(loc="upper left", fontsize=12)
price_ax.set_ylabel("Price")
price_ax.set_title("HSBA.L Daily Price", fontsize=24)

# ROC subplot
roc_ax = plt.subplot(2, 1, 2, sharex = price_ax)
roc_ax.plot(roc, color='k', linewidth = 1, alpha=0.7, label="9-Day ROC")
roc_ax.legend(loc="upper left", fontsize=12)
roc_ax.set_ylabel("% ROC")

# Adding a horizontal line at the zero level in the ROC subplot:
roc_ax.axhline(0, color = (.5, .5, .5), linestyle = '--', alpha = 0.5)

# Filling the areas between the indicator and the level 0 line:
roc_ax.fill_between(dates, 0, roc, where = (roc >= 0), color='g', alpha=0.3, interpolate=True)
roc_ax.fill_between(dates, 0, roc, where = (roc  < 0), color='r', alpha=0.3, interpolate=True)

# Formatting the date labels
roc_ax.xaxis.set_major_formatter(DateFormatter('%b'))

# Formatting the labels on the y axis for ROC:
roc_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())

# Adding a grid to both subplots:
price_ax.grid(color='grey', linestyle='-', alpha=0.5)
roc_ax.grid(color='grey', linestyle='-', alpha=0.5)

# Setting a background color for the both subplots:
price_ax.set_facecolor((.94,.95,.98))
roc_ax.set_facecolor((.98,.97,.93))

# Adding margins around the plots:
price_ax.margins(0.05, 0.2)
roc_ax.margins(0.05, 0.2)

# Hiding the tick marks from the horizontal and vertical axis:
price_ax.tick_params(left=False, bottom=False)
roc_ax.tick_params(left=False, bottom=False, labelrotation=45)

# Hiding all the spines for the price subplot:
for s in price_ax.spines.values():
    s.set_visible(False)
# Hiding all the spines for the ROC subplot:
for s in roc_ax.spines.values():
    s.set_visible(False)

# To better separate the two subplots, we reinstate a spine in between them
roc_ax.spines['top'].set_visible(True)
roc_ax.spines['top'].set_linewidth(1.5)

st.pyplot(fig)

st.write('Now let us plot some more graphs for the same period to get more insights from the data: a candlestick plot and a volume plot.')

st.pyplot(mpf.plot(hsba_roc_100d, type='candle',  style='yahoo', figsize=(15,8),  title="HSBA.L Daily Price", volume=True))
