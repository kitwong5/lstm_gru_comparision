import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import forxpy
from forxpy.forxpy import *
from forex_python.converter import CurrencyRates
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, GRU, Dense, Input, Activation, concatenate
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import MetricFrame
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

tf.random.set_seed(20)
np.random.seed(10)

def run_prediction(ticker_symbol, model_name, data_source):
     
    ## get data
    if data_source == 'STOCK':
        # 1) yfinance
        data=yf.download(ticker_symbol, start='2017-01-03', end='2022-12-31')[['Adj Close']]
    else:
        # 2) forxpy
        fx_df = retrieve_data(export_csv = False)
        fx_df['USD_AUD'] = fx_df['USD']/fx_df['AUD']
        fx_df['USD_HKD'] = fx_df['USD']/fx_df['HKD']
        fx_df['HKD_AUD'] = fx_df['HKD']/fx_df['AUD']
        data= fx_df[['date','HKD_AUD']]    
        data.drop('date', axis=1, inplace=True)

    print(data)
    
    ##split data to training and test data
    split_percentage=0.7
    split_point=round(len(data)*split_percentage)
    train_data=data.iloc[:split_point]
    test_data=data.iloc[split_point:]
    
    ## normalize data
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)

    X_train, Y_train, X_test,Y_test=timeseries_preprocessing(scaled_train, scaled_test, 10)    
    model = Sequential()
    
    if model_name == 'LSTM':
        #LSTM
        model.add(LSTM(288,input_shape=(X_train.shape[1],1)))
        model.add(Dense(1))
    else:
        #GRU
        model.add(GRU(288,input_shape=(X_train.shape[1],1), activation='tanh'))  #activation='sigmoid'
        model.add(Dense(1))

    model.compile(optimizer='adam',loss='mse')

    print(model.summary())
    
    history = model.fit(x=X_train,y=Y_train,epochs=5,validation_data=(X_test,Y_test),shuffle=False)
    
    axes=plt.axes()
    axes.plot(pd.DataFrame(model.history.history)['loss'], label='Loss')
    axes.plot(pd.DataFrame(model.history.history)['val_loss'], label='Validation Loss')
    axes.legend(loc=0)
    axes.set_title('Model fitting performance')
    plt.show()
    Y_predicted=scaler.inverse_transform(model.predict(X_test))
    Y_true=scaler.inverse_transform(Y_test.reshape(Y_test.shape[0],1))
    axes=plt.axes()
    axes.plot(Y_true, label='True Y')
    axes.plot(Y_predicted, label='Predicted Y')
    axes.legend(loc=0)
    axes.set_title('Prediction adjustment')
    plt.show()

    print('Model accuracy (%)')
    Y_p=scaler.inverse_transform(model.predict(X_train))
    Y_t=scaler.inverse_transform(Y_train.reshape(Y_train.shape[0],1))
    print((1-(metrics.mean_absolute_error(Y_t, Y_p)/Y_t.mean()))*100)
    print('')
    print('Prediction performance')
    print('MAE in %', (metrics.mean_absolute_error(Y_true, Y_predicted)/Y_true.mean())*100)
    print('MSE', metrics.mean_squared_error(Y_true, Y_predicted))
    print('RMSE',np.sqrt(metrics.mean_squared_error(Y_true, Y_predicted)))
    print('R2', metrics.r2_score(Y_true, Y_predicted))
    
def timeseries_preprocessing(scaled_train, scaled_test, lags):

    X,Y = [],[]
    for t in range(len(scaled_train)-lags-1):
        X.append(scaled_train[t:(t+lags),0])
        Y.append(scaled_train[(t+lags),0])
    
    Z,W = [],[]
    for t in range(len(scaled_test)-lags-1):
        Z.append(scaled_test[t:(t+lags),0])
        W.append(scaled_test[(t+lags),0])
        
    X_train, Y_train, X_test, Y_test=np.array(X), np.array(Y), np.array(Z),np.array(W)

    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
    
    return X_train, Y_train, X_test, Y_test



run_prediction('DJI', 'LSTM', 'STOCK') #DOW JONES INDICES, LSTM model, Stock Price prediction
#run_prediction('DJI', 'GRU', 'STOCK') #DOW JONES INDICES, GRU model, Stock Price prediction
#run_prediction('DJI', 'LSTM', 'FX') #DOW JONES INDICES, LSTM model, FX Price prediction
#run_prediction('DJI', 'GRU', 'FX') #DOW JONES INDICES, GRU model, FX Price prediction