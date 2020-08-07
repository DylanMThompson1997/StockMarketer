"""
Dylan Thompson
03-06-2020

Python Share Price Predictions
Using YAHOO FINANCE CSV FILES

Needed installs: lxml, yfinance, pandas, skilearn, numpy
"""

import yfinance as yf
import numpy as np 
from matplotlib import pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

def main():
    list_of_companies = ["SKC.NZ", "MSFT", "AAPL", "GOOGL", "CNU.NZ", "MET", "MFT.NZ", "AMC.AX", "RHC.AX", "AIA.NZ", "VCT.NZ", "EBO.NZ", "CNU.AX"]
    list_of_indices = ["^NZ50", "^IXIC", "^FTSE", "000001.SS", "LSE.L"]

    j = create_ticker("JPEX")
    four_plot_summary(j, "12mo")
    price_predict(j, "max")

def getName(ticker):
    return ticker.ticker

def create_ticker(company_symbol):
    return yf.Ticker(company_symbol)

def price_predict(company, interval):
    data = company.history(period=interval)
    #print('Raw data from Yahoo Finance : ')
    #print(data.head())
    data = data.drop('Dividends',axis=1) 
    data = data.drop('Stock Splits',axis = 1)
    #print('\n\nData after removing : ')
    #print(data.head())
        # Split into train and test data
    data_X = data.loc[:,data.columns !=  'Close' ]
    data_Y = data['Close']
    train_X, test_X, train_y,test_y = train_test_split(data_X,data_Y,test_size=0.25)
    #print('\n\nTraining Set')
    #print(train_X.head())
    #print(train_y.head())
    regressor = LinearRegression()
    regressor.fit(train_X,train_y)
    predict_y = regressor.predict(test_X)
    #print('Prediction Score : ' , regressor.score(test_X,test_y))
    error = mean_squared_error(test_y,predict_y)
    #print('Mean Squared Error : ',error)
    fig = plt.figure()
    ax = plt.axes()
    ax.grid()
    ax.set(xlabel='Close ($)',ylabel='Open ($)', title='{0} Stock Prediction using Linear Regression'.format(company.ticker))
    ax.plot(test_X['Open'],test_y, label="Test Set")
    ax.plot(test_X['Open'],predict_y, label="Prediction Set")
    ax.legend()
    #fig.savefig('LRPlot.png')
    plt.show()

def price_comp_graph(company, interval):
    p_open = company.history(period=interval)["Open"]
    p_close = company.history(period=interval)["Close"]
    y = p_close - p_open
    plt.title("Price Gains Over Time for {0}".format(company.ticker)) 
    plt.xlabel("Dates") 
    plt.ylabel("Close - Open Prices") 
    plt.plot(y) 
    plt.show()

def closing_price_graph(company, interval):
    p_close = company.history(period=interval)["Close"]
    y = p_close
    plt.title("Closing Prices Over Time for {0}".format(company.ticker)) 
    plt.xlabel("Dates") 
    plt.ylabel("Closing Prices") 
    plt.plot(y) 
    plt.show()

def price_box_whisker(comp_arr, interval):
    fig, axs = plt.subplots(len(comp_arr)//2, 2)
    for i in comp_arr:
        price = i.history(period=interval)["Close"]        
        if comp_arr.index(i) % 2 == 0: 
            axs[comp_arr.index(i)//2][0].set_ylabel(i)
            axs[comp_arr.index(i)//2][0].boxplot(price)
        else:
            axs[comp_arr.index(i)//2][1].set_ylabel(i)
            axs[comp_arr.index(i)//2][1].boxplot(price)
    fig.show()

def price_diff_closing_price(comp_arr, interval):
    fig, axs = plt.subplots(len(comp_arr), 2)
    plt.xticks(rotation='vertical')
    axs[0][0].set_title("Price Change")
    axs[0][1].set_title("Closing Prices")
    for i in comp_arr:
        open_price = i.history(period=interval)["Open"]
        close_price = i.history(period=interval)["Close"]
        axs[comp_arr.index(i)][0].set_ylabel(i)
        axs[comp_arr.index(i)][0].plot(close_price - open_price, color='red', linewidth=0.5)
        axs[comp_arr.index(i)][1].plot(close_price, color='blue', linewidth=0.5)
        if i != comp_arr[-1]:
            axs[comp_arr.index(i)][0].axes.get_xaxis().set_visible(False)
            axs[comp_arr.index(i)][1].axes.get_xaxis().set_visible(False)
        else:
            continue
    #plt.xticks(rotation=90)
    fig.subplots_adjust(wspace=0)
    fig.show()

def four_plot_summary(company, interval):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('{0} Summary Data'.format(company.ticker), fontsize=12)
    close_price = company.history(period=interval)["Close"]
    open_price = company.history(period=interval)["Open"]
    axs[0][0].plot(close_price - open_price, color='red', linewidth=0.5)
    axs[1][0].plot(close_price, color='blue', linewidth=0.5)
    axs[0][1].plot(close_price, open_price, color='green', linewidth=0.5)
    axs[1][1].boxplot([open_price, close_price])
    fig.savefig('CompPlot{0}.png'.format(company.ticker))
    fig.show()

main()
