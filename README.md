


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


## Step 2. Chart patterns and technical indicators (WIP)
As stated above: Technical analysis is the use of charts and technical indicators to identify trading signals and price patterns so an investigative attempt will be made using the most common indicators.  

## Step 3. Hypothesis testing and validation (WIP)
Based on the fact that historically FTSE 100 index has demonstrated drastic volatility changes a hypothesis can be constructed and tested: does this volatility signify any fundamental changes and therefore likely to continue or it could just happen by chance.

## Step 4. Dashboards build (WIP)
An exercise in building a comprehensive trading dashboard. For the said exercise the same selection of FTSE 100 companies equity is taken  as well as a single company stock: HSBC (HSBA.L) 

## Adapting the pipeline
These four jupyter books should be considered as a pipeline/framework. Please download them, test and adapt to your specific needs. Various stocks or time periods can be taken and explored. 

The `.py` files going along them are Streamlit deployments of the said jupyter projects.


PS: This project has been undertaken as an attempt to bring to order a total zoo of ideas and concepts gained from self-education on the intricacies of the stock-market, so an overwhelming amount of comments in the code and web-version is begging to be excused. 

PPS: Acknowledgments: to the one and only Alison Mitchell for her work and explanations.
