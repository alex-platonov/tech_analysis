


# Technical analysis
This  an excercise in technical analysis wrapped up in a catchy web-app form (thanks to Streamlit).

**Wikipedia-like definition**: Technical analysis is the use of charts and technical indicators to identify trading signals and price patterns. 

An attempt to investigate various strategies such as :
 - leading and lagging trend;
 - momentum;
 
Apart from that, various volatility and volume indicators will be explored: 
- Moving Averages; 
- Moving Average Convergence Divergence (MACD); 
- Stochastic Oscillator; 
- Relative Strength Index (RSI);
- Money Flow Index (MFI); 
- Rate of Change (ROC); 
- Bollinger Bands;
- On-Balance Volume (OBV).

## Step 1. Data collection and Exploratory Data Analysis (EDA)
Historical data on several FTSE 100 index companies will be collected, analyzed, and visualized in an attempt to gain insights into their equity market performance from 2014 to 2024. The market behavior of the index itself will also be analyzed. 

Live demo is available here: https://techanalysis-8zz9ru24q8kcvnue2tuh4f.streamlit.app/

A relatively safe, however representative list of stocks has been identified (huge thanks to Alison Mitchell):
- ULVR.L (Unilever)
- SHEL.L (Shell)
- GSK.L (GlaxoSmithKline)
- AZN.L (AstraZeneca)
- HSBA.L (HSBC)
- BP.L (BP) 

This list represents a selection of different industries, namely - pharmaceuticals, oil, and finance. 

![Step_1](https://github.com/alex-platonov/tech_analysis/blob/main/step_1.gif)

*** 
## Step 2. Chart patterns and technical indicators 
As stated above: Technical analysis is the use of charts and technical indicators to identify trading signals and price patterns so an investigative attempt will be made using the most common indicators.  
UPDATE: so it took me over 2 months to whip this one into shape :) 
UPDATE2: unfortunately this notebook seems to be too heavy for Streamlit cloud implementation. If you still wish to make it interactive - please use 02_patterns_and_indicators.py (it is already preconfigured for Streamlit rollout)

For this exercise we shall be taking the same FTSE100 constituent stocks: AZN.L GSK.L ULVR.L BP.L SHEL.L HSBA.L, however, the main honorable Guinea Pig will still be HSBA.L.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/c7f3ee08-7f94-46b9-8cf3-0a0eeb69868f)

Then we shall attempt to discover a whole plethora of technical strategies: 

### Moving averages:
Moving averages smooth a series filtering out noise to help identify trends, one of the fundamental principles of technical analysis being that prices move in trends. Types of moving averages include simple, exponential, smoothed, linear-weighted, MACD, and as lagging indicators they follow the price action and are commonly referred to as trend-following indicators.

#### Simple Moving Average (SMA):
The simplest form of a moving average, known as a Simple Moving Average (SMA), is calculated by taking the arithmetic mean of a given set of values over a set time period. This model is probably the most naive approach to time series modeling and simply states that the next observation is the mean of all past observations and each value in the time period carries equal weight.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/c350df5f-0355-4a6c-9b42-ec6d7ac1d547)

#### Moving Average Crossover Strategy: 
The most popular moving average crossover strategy, and the "Hello World!" of quantitative trading, being the easiest to construct, is based on the simple moving average. When moving averages cross, it is usually confirmation of a change in the prevailing trend, and we want to test whether over the long term the lag caused by the moving average can still give us profitable trades.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/71da0e85-3d79-4ee0-b6ae-0af1ba52b984)

#### Exponential Moving Average
In a Simple Moving Average, each value in the time period carries equal weight, and values outside of the time period are not included in the average. However, the Exponential Moving Average is a cumulative calculation where a different decreasing weight is assigned to each observation. Past values have a diminishing contribution to the average, while more recent values have a greater contribution. This method allows the moving average to be more responsive to changes in the data.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/799f0144-76d9-4b05-8cbc-a41798a84306)

#### Triple Moving Average Crossover Strategy
This strategy uses three moving averages - short/fast, middle/medium and long/slow - and has two buy and sell signals.
The first is to buy when the middle/medium moving average crosses above the long/slow moving average and the short/fast moving average crosses above the middle/medium  # moving average. If we use this buy signal the strategy is to sell if the short/fast moving average crosses below the middle/medium moving average.
The second is to buy when the middle/medium moving average crosses below the long/slow moving average and the short/fast moving average crosses below the middle/medium moving average. If we use this buy signal the strategy is to sell if the short/fast moving average crosses above the middle/medium moving average.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/9e1c0666-a9d0-43f1-bf2a-86fc012bdb76)

#### Exponential Smoothing
Single Exponential Smoothing, also known as Simple Exponential Smoothing, is a time series forecasting method for univariate data without a trend or seasonality. It requires an alpha parameter, also called the smoothing factor or smoothing coefficient, to control the rate at which the influence of the observations at prior time steps decay exponentially.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/d3a17c5f-51a0-4298-95d7-5a1e3844f327)

#### Double Exponential Smoothing
aka Holt’s Linear Trend Model) is an extension of the previous one being a recursive use of Exponential Smoothing twice where beta is the trend smoothing factor, and takes values between 0 and 1. It explicitly adds support for trends.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/abba22c5-3b9d-4fe7-8767-7fe42d89df2c)

#### Moving average convergence divergence (MACD)
The MACD is a trend-following momentum indicator that turns two trend-following indicators, moving averages, into a momentum oscillator by subtracting the longer moving average from the shorter one.
It is useful although lacking one prediction element - because it is unbounded it is not particularly useful for identifying overbought and oversold levels. Traders can look for signal line crossovers, neutral/centreline crossovers (otherwise known as the 50 level), and divergences from the price action to generate signals.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/a2482281-91e4-4eef-8a10-acab9ad8c891)

### Momentum Strategies:
In momentum trading strategies stocks have momentum (i.e. upward or downward trends) that we can detect and exploit.

#### Relative Strength Index (RSI)
The RSI is a momentum indicator. A typical momentum strategy will buy stocks that have been showing an upward trend in hopes that the trend will continue, and make predictions based on whether the past recent values were going up or going down.
The RSI determines the level of overbought (70) and oversold (30) zones using a default lookback period of 14 i.e. it uses the last 14 values to calculate its values. The idea is to buy when the RSI touches the 30 barrier and sell when it touches the 70 barrier.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/cef8f32b-37cf-4597-950d-d11d2ff6998d)

### Money Flow Index (MFI)
Money Flow Index (MFI) is a technical oscillator, and momentum indicator, that uses price and volume data for identifying overbought or oversold signals in an asset. It can also be used to spot divergences that warns of a trend change in price. The oscillator moves between 0 and 100 and a reading of above 80 implies overbought conditions, and below 20 implies oversold conditions.
It is related to the Relative Strength Index (RSI) but incorporates volume, whereas the RSI only considers price.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/78e50db9-8a44-41c8-9ef9-b814277c2c82)

#### Stochastic Oscillator
The stochastic oscillator is a momentum indicator comparing the closing price of a security to the range of its prices over a certain period of time and is one of the best-known momentum indicators along with RSI and MACD.
The intuition is that in a market trending upward, prices will close near the high, and in a market trending downward, prices close near the low. The stochastic oscillator is plotted within a range of zero and 100. The default parameters are an overbought zone of 80, an oversold zone of 20 and well-used lookback period of 14 and 5 which can be used simultaneously. The oscillator has two lines, the %K and %D, where the former measures momentum and the latter measures the moving average of the former. The %D line is more important of the two indicators and tends to produce better trading signals which are created when the %K crosses through the %D.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/c7207d86-1f84-4e1d-98c1-3508a0e95b21)

#### Rate of Change (ROC)
The ROC indicator is a pure momentum oscillator. The ROC calculation compares the current price with the price "n" periods ago e.g. when we compute the ROC of the daily price with a 9-day lag, we are simply looking at how much, in percentage, the price has gone up (or down) compared to 9 days ago. Like other momentum indicators, ROC has overbought and oversold zones that may be adjusted according to market conditions.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/13a4b9b1-8ba1-4500-9834-fac349bcfe43)

### Volatility trading strategies:
Volatility trading involves predicting the stability of an asset’s value. Instead of trading on the price rising or falling, traders take a position on whether it will move in any direction.

#### Bollinger Bands
A Bollinger Band is a volatility indicator based on based on the correlation between the normal distribution and stock price and can be used to draw support and resistance curves. It is defined by a set of lines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of the security's price, but can be adjusted to user preferences.
By default, it calculates a 20-period SMA (the middle band), an upper band two standard deviations above the moving average, and a lower band two standard deviations below it.
If the price moves above the upper band this could indicate a good time to sell, and if it moves below the lower band it could be a good time to buy.
Whereas the RSI can only be used as a confirming factor inside a ranging market, not a trending market, by using Bollinger bands we can calculate the widening variable,
or moving spread between the upper and the lower bands, which tells us if prices are about to trend and whether the RSI signals might not be that reliable.
Despite 90% of the price action happening between the bands, however, a breakout is not necessarily a trading signal as it provides no clue as to the direction
and the extent of future price movement.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/d1397458-09ee-47e2-be13-1d69c4fbc877)

### Mean reversion strategies:
In mean reversion algorithmic trading strategies stocks return to their mean and we can exploit when it deviates from that mean.

#### Pairs Trading
Pairs Trading is the Holy Grail of market neutral strategies and a type of statistical arbitrage, which is exploiting statistical properties that we believe can make money if they continue.
The basic idea is to select two cointegrated stocks which move similarly/are statistically related and deviate from their mean. Then sell the high priced stock and buy the low priced stock where there is a price divergence between the pairs. You make money from a pairs trade when your “long” outperforms your “short”: If your “long” rises more than your “short” or if your “long” falls less than your “short”.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/687c37e1-c0da-44c5-80f0-66e910c1cf2a)

### Volume Trading Strategies:
Volume trading is a measure of how much of a given financial asset has traded in a period of time. Volume traders look for instances of increased buying or selling orders. They also pay attention to current price trends and potential price movements. Generally, increased trading volume will lean heavily towards buy orders.

#### On Balance Volume (OBV)
OBV is a momentum-based indicator which measures volume flow to gauge the direction of the trend. Volume and price rise are directly proportional and OBV can be used as a confirmation tool with regards to price trends. A rising price is depicted by a rising OBV and a falling OBV stands for a falling price.

![image](https://github.com/alex-platonov/tech_analysis/assets/154932143/7c6d92cd-5bdc-4488-81c4-a2222c67def7)

***

## Step 3. Hypothesis testing and validation (WIP)
Based on the fact that historically FTSE 100 index has demonstrated drastic volatility changes a hypothesis can be constructed and tested: does this volatility signify any fundamental changes and therefore likely to continue or it could just happen by chance.

## Step 4. Dashboards build (WIP)
An exercise in building a comprehensive trading dashboard. For the said exercise the same selection of FTSE 100 companies equity is taken  as well as a single company stock: HSBC (HSBA.L) 

## Adapting the pipeline
These four jupyter books should be considered as a pipeline/framework. Please download them, test and adapt to your specific needs. Various stocks or time periods can be taken and explored. 

The `.py` files going along them are Streamlit deployments of the said jupyter projects.


PS: This project has been undertaken as an attempt to bring to order a total zoo of ideas and concepts gained from self-education on the intricacies of the stock-market, so an overwhelming amount of comments in the code and web-version is begging to be excused:) 

PPS: Acknowledgments: to the one and only Alison Mitchell for her work and explanations.
