import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.patches as mpatches
from matplotlib.dates import date2num, DateFormatter, WeekdayLocator, DayLocator, MONDAY
from mplfinance.original_flavor import candlestick_ohlc
import seaborn as sns
import streamlit as st
import datetime
from datetime import date, timedelta
from io import StringIO
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%matplotlib inline

st.title('FTSE_100 Technical analysis')
st.subheader('Step 1: data collection and exploratory analysis')
st.image('https://raw.githubusercontent.com/alex-platonov/tech_analysis/4664aa3f0f0ed1778c93722b75de26cdc2f14a93/technical-analysis.jpg')

st.markdown("<hr>", unsafe_allow_html=True)

st.write('This is an excercise in technical analysis wrapped up in a catchy web-app form (thanks to Streamlit). The present page (WIP) is step #1 in a 4-step analysis pipeline')

st.write('Historical data on several FTSE 100 index constituent companies will be collected, analyzed, and visualized in an attempt to gain insights into their equity market performance from 2014 to 2024. The market behavior of the index itself will also be analyzed.')
text = """
A relatively safe, however representative list of stocks has been identified (huge thanks to Alison Mitchell):

- ULVR.L (Unilever)
- SHEL.L (Shell)
- GSK.L (GlaxoSmithKline)
- AZN.L (AstraZeneca)
- HSBA.L (HSBC)
- BP.L (BP)

This list represents a selection of different industries, namely - pharmaceuticals, oil, and finance that will no doubt contribute to the overall statistical representativeness.
"""
st.write(text)

st.subheader('Data collection')

ftse100_stocks = yf.download("AZN.L GSK.L ULVR.L BP.L SHEL.L HSBA.L", start=datetime.datetime(2014, 1, 1), 
                                     end=datetime.datetime(2023, 12, 31), group_by='tickers')
ftse100_stocks.head(10)
st.dataframe(ftse100_stocks.head(10))
#------------------------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.subheader('Data exploration')
st.write('Let us attempt some data exploration')
ftse100_stocks.describe()
description = ftse100_stocks.describe()
st.dataframe(description)
#------------------------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Data check')
st.write('Summarising data to a dataframe to see if any values of datatypes are missing')
buffer = StringIO()
ftse100_stocks.info(buf=buffer)
info_str = buffer.getvalue()

# Displaying the info string using markdown for better formatting
st.markdown("```\n" + info_str + "\n```")
#------------------------------------------------------------------------------------------------------------------------------------

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Trading days in the dataset')
st.write('Number of rows represents the number of trading days')
ftse100_stocks.shape
#st.dataframe(ftse100_stocks.shape)

#------------------------------------------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Adjusted Close Price')
st.write('Adjusted Close price will be used to take into account all corporate actions, such as stock splits and dividends, to give a more accurate reflection of the true value of the stock and present a coherent picture of returns.')

adj_close = pd.DataFrame()

# List of tickers
tickers = ['AZN.L', 'GSK.L', 'ULVR.L', 'BP.L', 'SHEL.L', 'HSBA.L']

# Extracting Adjusted Close prices for each ticker
for ticker in tickers:
    adj_close[ticker] = ftse100_stocks[ticker]['Adj Close']

# Display the DataFrame in Streamlit
st.dataframe(adj_close)

# ------------------------------------------------------------------------------------------------------------------------------------

# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Adjusted Close price for each company stock.') 

# Initialize an empty DataFrame for adjusted close prices
adj_close = pd.DataFrame()

# List of tickers
tickers = ['AZN.L', 'GSK.L', 'ULVR.L', 'BP.L', 'SHEL.L', 'HSBA.L']

# Extracting Adjusted Close prices
for ticker in tickers:
    adj_close[ticker] = ftse100_stocks[ticker]['Adj Close']

# Set the plot size (optional)
sns.set(rc={'figure.figsize':(15, 9)})

# Plotting
plt.figure(figsize=(15, 9))
adj_close.plot(grid=True)
plt.title('Adjusted Close Price for all stocks', color='black', fontsize=20)
plt.xlabel('Year', color='black', fontsize=15)
plt.ylabel('Adjusted Close Price (pence)', color='black', fontsize=15)

# Display the plot in Streamlit
st.pyplot(plt)

st.write('Here the absolute price is displayed rather than relative change which we are more concerned with when trading. AZN.L and ULVR.L stocks are far more expensive than BP.L and HSBA.L making the latter appear much less volatile than they truly are.')

#--------------------------------------------------------------------------------------------------------------------------------------

# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Minimum and maximum')
st.write('Lets alculate min and max Adjusted Close price')

adj_close_min_max = adj_close.apply(lambda x: pd.Series([x.min(), x.max()], 
                              index=['min', 'max']))

st.dataframe(adj_close_min_max)
#--------------------------------------------------------------------------------------------------------------------------------------

# Divider
st.markdown("<hr>", unsafe_allow_html=True) 
st.subheader('Adjusted Close Price with two different scales')
st.write('Let us plot BP.L and HSBA.L data on a secondary y-axis')

# Set the plot size and style using seaborn
sns.set(rc={'figure.figsize':(15, 9)})

# Plotting the DataFrame with a secondary Y-axis for specific tickers
plt.figure(figsize=(15, 9))
ax = adj_close.plot(secondary_y=["BP.L", "HSBA.L"], grid=True)
plt.title('Adjusted Close Price with two different scales', color='black', fontsize=20)
plt.show()

# Display the plot in Streamlit
st.pyplot(plt)
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True) 
st.subheader('Returns of each stock')
st.markdown("""
Next, we would want to see the returns of each of the stocks.
However, this requires transforming of the data to better suit our needs. 
So, we would want to plot $\(return_{t,0} = \frac{price_t}{price_0}$ by applying the lambda function to each column in an adjusted close dataframe.
""", unsafe_allow_html=True)

returns_lambda = adj_close.apply(lambda x: x / x[0])
returns_lambda.head()
st.dataframe(returns_lambda.head())

#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Profitability of each stock')
st.write('Plot return_{t,0}  = \frac{price_t}{price_0} with transformed data to get an insight on how profitable the stock had been.')

returns_lambda.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
sns.set(rc={'figure.figsize':(15, 9)})
plt.title('Stock returns for 10 year time period', color = 'black', fontsize = 20)
plt.xlabel('Year', color = 'black', fontsize = 15)
plt.ylabel('Returns (%)', color = 'black', fontsize = 15);
st.pyplot(plt)
st.write('Covid lockdown alterations could be clearly observed: HSBC for example plummeted down and had not recovered until late 2022, when AstraZeneca soared at the beginning of 2020. So we can clearly state that such plot is way more useful as we can observe not only the profitability of each stock, but also the correlation of some stocks (especially in the year 2020)')

#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
#st.markdown("<hr>", unsafe_allow_html=True)
#st.subheader('Change in profitability per day')
#st.write('Now let us create a dataframe to contain returns for each company stock to plot the change of each stock per day')
returns = pd.DataFrame()
#st.dataframe(pd.DataFrame())

#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Change in profitability per day in %')
st.write('Another point of interest is the daily volatility (aka percentage change) with the formula increase_{t}  = \frac{price_t - price_{t-1}}{price_t}. This can be achieved with the pandas  pct_change() method which computes the percentage change from the previous row by default.')

tickers = ['AZN.L', 'GSK.L', 'ULVR.L', 'BP.L', 'SHEL.L', 'HSBA.L']
for ticker in tickers:
    returns[ticker] = ftse100_stocks[ticker]['Adj Close'].pct_change() * 100
st.dataframe(returns)

#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Cleaning the data')
st.write('Cleaning the data by dropping the NaN values')
returns.dropna(inplace=True)
returns.head()
st.dataframe(returns.head())
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Stock reeturns for 2023')
st.write('Plot returns for 2023 that will show changes between trading days. THis is generally considered as a more advanced approach to modelling of equity behaviour.')

returns.loc['2023-01-01':'2023-12-31'].plot(grid = True).axhline(y = 1, color = "black", lw = 2)
sns.set(rc={'figure.figsize':(15, 9)})
plt.title('Stock returns for 2023', color = 'black', fontsize = 20)
plt.xlabel('Date', color = 'black', fontsize = 15)
plt.ylabel('Returns (%)', color = 'black', fontsize = 15);
st.pyplot(plt)

#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Log differences of the adjusted price data')
text = """
Another way to explore stock growth is with log differences. Taking the natural log of the prices will give an approximation to the true daily returns.
This can be represented with the formula 𝑐ℎ𝑎𝑛𝑔𝑒𝑡  =  𝑙𝑜𝑔(𝑝𝑟𝑖𝑐𝑒𝑡)−𝑙𝑜𝑔(𝑝𝑟𝑖𝑐𝑒𝑡−1)
According to the Financial math textbook: 
Using logs, or summarising changes in terms of continuous compounding, has a number of advantages over looking at simple percent changes. 
For example, if your portfolio goes up by 50% (say from £100 to £150) and then declines by 50% (say from £150 to £75), you’re not back where you started. 
If you calculate your average percentage return (in this case, 0%), that’s not a particularly useful summary of the fact that you actually ended up 25% below where you started.
By contrast, if your portfolio goes up in logarithmic terms by 0.5, and then falls in logarithmic terms by 0.5, you are exactly back where you started. 
The average log return on your portfolio is exactly the same number as the change in log price between the time you bought it and the time you sold it, 
divided by the number of years that you held it.
"""
st.write(text)
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.write('Use numpy log function to obtain and plot the log differences of the adjusted price data')
stock_change = adj_close.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
stock_change.head()
st.dataframe(stock_change.head())

#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.write('Clean up the data by dropping NaN values')

stock_change.dropna(inplace=True)
stock_change.head()
st.dataframe(stock_change.head())

#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.write('Plot log differences for 2014-2024')
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
sns.set(rc={'figure.figsize':(15, 9)})
plt.title('Log differences of stocks for 10 year time period', color = 'black', fontsize = 20)
plt.xlabel('Year', color = 'black', fontsize = 15)
plt.ylabel('Natural log', color = 'black', fontsize = 15);
st.pyplot(plt)
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.write('Plot log differences for 2023 (since 2024 has just begun)')
stock_change.loc['2023-01-01':'2023-12-31'][1:].plot(grid = True).axhline(y = 0, color = "black", lw = 2)
sns.set(rc={'figure.figsize':(15, 9)})
plt.title('Log differences of stocks for 2023', color = 'black', fontsize = 20)
plt.xlabel('Year', color = 'black', fontsize = 15)
plt.ylabel('Natural log', color = 'black', fontsize = 15);
st.pyplot(plt)

#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.write('To keep the returns on the same time scale - the annual percentage rate needs to be computed')
stock_change_apr = stock_change * 252 * 100    # There are 252 trading days in a year; the 100 converts to percentages
st.dataframe(stock_change_apr)
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.write('Plotting annualised returns for the last year (2023)')
stock_change_apr['2023-01-01':'2023-12-31'].plot(grid = True).axhline(y = 0, color = "black", lw = 2)
sns.set(rc={'figure.figsize':(15, 9)})
plt.title('Annual percentage rate (APR) for 2023', color = 'black', fontsize = 20)
plt.xlabel('Date', color = 'black', fontsize = 15)
plt.ylabel('APR', color = 'black', fontsize = 15);
st.pyplot(plt)
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.write('Worst single day returns')
returns.idxmin()
st.dataframe(returns.idxmin())
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.write('Best single day returns')
returns.idxmax()
st.dataframe(returns.idxmax())
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Mean')
st.write('The mean is used to estimate the performance of a company stock price over a particular time period. Here it is the average of the returns, and also determines the standard deviation and variance.')
st.write('Computing mean to give a representation of the average expected returns')
returns.mean()
st.dataframe(returns.mean())
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Variance')
st.write('Variance measures variability from the average or mean. It correlates to the size of the overall range of the data set, being greater when there is a wider range and narrower when there is a narrower range. It is calculated by taking the differences between each value in the data set and the mean, squaring the differences to make them positive, and dividing the sum of the squares by the number of values in the data set. The calculation of variance uses squares because it weighs outliers more heavily than data closer to the mean. This calculation also prevents differences above the mean from cancelling out those below, which would result in a variance of zero. Variance formula σ<sup>2</sup> = \frac {\sum_{i = 1}^n (x_i - \overline{x})^2}{n}')
st.write('Computing variance to give a measure of the dispersion of returns around the mean')
returns.var()
st.dataframe(returns.var())
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Standard deviation')
st.write('Standard deviation (σ) is often used by investors to measure the risk of a stock or a stock portfolio, the basic idea being that it is a measure of volatility. It looks at how far from the mean a group of values is, and is calculated as the square root of variance by figuring out the variation between each data point relative to the mean. Essentially, it is the square root of the average squared deviation from the mean, and the more spread out the values are, the higher the standard deviation.')
st.write('Computing the standard deviation to describe variability in the stock returns from the mean')
returns.std()
st.dataframe(returns.std())
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Skewness')
st.write('Financial returns are typically positively or negatively skewed and warp the look of the normally distributed bell-shaped curve distorting the accuracy of standard deviation as a measure of risk. Skewness essentially measures the relative size of the two tails of the distribution.')
st.write('Computing skewness to measure the asymmetry of the data around its mean')
returns.skew()
st.dataframe(returns.skew())
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Kurtosis')
st.write('Kurtosis is a measure of the combined sizes of the two tails - not the peakedness or flatness. It measures the tail-heaviness of the distribution, or amount of probability in the tails.')
st.write('Computing kurtosis as a measure of the combined sizes of the two tails.')
returns.kurt()
st.dataframe(returns.kurt())

st.write('A normal distribution has a kurtosis of 3, however the pandas kurtosis function makes it a uniform zero and in this case the measure is called excess kurtosis. Since the analysed decade had both calm years at the beginning and super-turbulent 2020/2021/2022/2023 nearly all of our honorable guinea pigs demonstrate excess kurtosis.')
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Data visualisation')
st.write('Plot pairwise relationships of the stocks with the pairplot() function which uses scatterplot() for each pairing of the variables and histplot() for the marginal plots along the diagonal.')
st.write('Pairplot of returns dataframe')
sns.pairplot(returns); # Plot the pairplot with Seaborn
sns.pairplot(returns.dropna())  # Using dropna() to remove NaN values for clean plotting
plt.tight_layout()  # Adjust subplots to fit into the figure area.
# Use st.pyplot() to display the plot
st.pyplot(plt.gcf())  # plt.gcf() gets the current figure (created by sns.pairplot)
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Boxplots')
text = """
Box and whisker plots are a standardised way of displaying the distribution of data based on a five number summary:
- minimum
- first quartile (Q1)
- median
- third quartile
- maximum
The line going through the box is the median.

Boxplots showing distribution of the returns data over the time period 
"""
st.write(text)

# Set the style for Seaborn plots
sns.set_style("whitegrid")

# Create a Matplotlib figure and axes
fig, axs = plt.subplots(ncols=6, nrows=1, figsize=(20, 10))
axs = axs.flatten()  # Flatten the axes array if needed

# Plotting each column in 'returns' using Seaborn's boxplot
index = 0
for k, v in returns.items():
    sns.boxplot(y=k, data=returns, ax=axs[index])
    index += 1

# Adjust layout
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

# Display the plot in Streamlit
st.pyplot(fig)

text = """
By comparing the interquartile ranges (box lengths), we can compare dispersion. If two boxes do not overlap with one another then there is a difference between the two groups. 
We can see that ULVR. does not overlap with the other stocks. If we compare the respective medians and the median line of one box lies outside of another entirely, then there is likely to be a difference between the two groups. Again we can see that ULVR. is different.

Whiskers show how big a range there is between maximum and minimum values, with larger ranges indicating wider distribution, that is, more scattered data. 
We can look for signs of skewness suggesting that data may not be normally distributed. Skewed data show a lopsided box plot, where the median cuts the box into two unequal pieces. 
If the longer part of the box is above the median, the data is said to be positively skewed. 
If the longer part is or below the median, the data is negatively skewed.

Any values in the data set that are more extreme than the adjacent values are plotted as separate points on the box plot. This identifies them as potential outliers.
"""
st.write(text)

#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Distribution plots')
st.write('Distribution plots depict the variation in the data distribution. Here the distribution of returns is shown by a histogram and a line in combination with it which is the kernel density estimate.')
st.write('Distribution plots showing the data for returns for 2023')

# Set the plotting style
sns.set_style("white")

# Define the tickers and prepare the figure and subplots
tickers = ['AZN.L', 'GSK.L', 'ULVR.L', 'BP.L', 'SHEL.L', 'HSBA.L']
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
axs = axs.flatten()

# Plotting histograms with a KDE for each ticker
for index, ticker in enumerate(tickers):
    # Filtering the 'returns' DataFrame for the specified date range before plotting
    sns.histplot(returns.loc['2023-01-01':'2023-12-31'][ticker], color='green', bins=100, ax=axs[index], kde=True)

# Adjust layout for better fit and readability
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)
#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Covariance')

st.write('Covariance indicates the direction of the linear relationship between variables. It is a measure of the relationship between two stocks returns and can help determine if stocks returns tend to move with or against each other. Investors might even be able to select stocks that complement each other in terms of price movement. This can help reduce the overall risk and increase the overall potential return of a portfolio.')
st.write('Covariance matrix to show direction of relationship between stocks returns')
st.dataframe(returns.cov()) 

#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Correlation')
st.write('Correlation matrix to show strength and direction of relationship between stocks returns')
st.dataframe(returns.corr())

st.write('The heatmap clearly shows the strength of correlation between pairs of company returns')

plt.figure(figsize=(10, 10))
sns.heatmap(data = returns.corr(), vmax=.8, linewidths=0.5,  fmt='.2f',
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.show()
st.pyplot(plt)

st.write('The strongest correlation can be observed between oil stocks (SHELL and BP), followed by the pharmaceutical stocks (GSK and AZN)') 

#--------------------------------------------------------------------------------------------------------------------------------------
# Divider
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('FTSE 100 Index data')
st.write('Finally let us get data from the launch of the FTSE 100 Index in January 1984 to the end of 2023 as being representative of the UK stock market.') 
# Download FTSE 100 historical stock data from Yahoo! Finance for 1984-2020
st.write('Download FTSE 100 historical stock data from Yahoo! Finance for 1984-2020')

ftse100_idx_to_2024 = yf.download("^FTSE", start=datetime.datetime(1984, 1, 1), 
                                     end=datetime.datetime(2024, 1, 1))
st.dataframe(ftse100_idx_to_2024) 

st.write('Now let us visualize the data')

def ftse100_to_2024_plot():
    plt.figure(figsize=(20, 10))  # Set the figure size for the plot
    sns.set_style("whitegrid")  # Set the Seaborn style
    
    # Plotting the 'Close' prices
    ftse100_idx_to_2024['Close'].plot(grid=True)
    
    # Highlight significant periods with red spans
    plt.axvspan('1987', '1989', color='r', alpha=0.5)
    plt.axvspan('2008', '2010', color='r', alpha=0.5)
    plt.axvspan('2020', '2024', color='r', alpha=0.5)
    
    # Adding a custom legend
    labs = mpatches.Patch(color='red', alpha=0.5, label="Black Monday, 2008 Crash, Covid-19 fall and its aftermath")
    plt.legend(handles=[labs], prop={"size":15}, bbox_to_anchor=(0.4, 0.1), loc='upper center', borderaxespad=0.)
    
    # Setting title and labels
    plt.title('Close Price for FTSE 100 stocks', color='black', fontsize=20)
    plt.xlabel('Year', color='black', fontsize=15)
    plt.ylabel('Close Price (pence)', color='black', fontsize=15)

# Call the function to plot
ftse100_to_2024_plot()

# Use st.pyplot() to display the plot in Streamlit
st.pyplot(plt.gcf())  # plt.gcf() gets the current figure

st.write('To be continued :)')

