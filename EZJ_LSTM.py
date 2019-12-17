from datetime import *
import pandas_datareader.data as data
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import math as m
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_dataset(dataset, look_back=1, look_forward=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


start = datetime(2005, 6, 1)
end = datetime.now()
EZJ = data.DataReader('EZJ.L', 'yahoo', start, end)

Train = EZJ.Close.values

scaler = MinMaxScaler(feature_range=(0, 1))
Train = scaler.fit_transform(Train[:, np.newaxis])

# split into train and test sets
train_size = int(len(Train) * 0.9)
test_size = len(Train) - train_size
train, test = Train[0:train_size,:], Train[train_size:len(Train),:]

trainPlot = np.empty_like(Train)
trainPlot[:, :] = np.nan
trainPlot[0:len(train):] = train
# shift test predictions for plotting
testPlot = np.empty_like(Train)
testPlot[:, :] = np.nan
testPlot[len(train):] = test
# plot baseline and predictions
plt.plot(scaler.inverse_transform(trainPlot))
plt.plot(scaler.inverse_transform(testPlot))
plt.show()

look_back = 32
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(100, input_shape=(1, look_back)))
model.add(Dense(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=1)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = m.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = m.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


trainPredictPlot = np.empty_like(Train)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(Train)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(Train)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(Train))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)

plt.xlim([len(Train)-testY.shape[1], len(Train)])
plt.ylim([700,1800])
plt.show()

error = np.mean(np.abs(testY-np.transpose(testPredict)))
stddev = np.std(np.abs(testY-np.transpose(testPredict)))
print(error, stddev)