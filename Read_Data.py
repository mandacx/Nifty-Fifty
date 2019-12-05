import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as m

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

owd = os.getcwd()
os.chdir('nifty50-stock-market-data')
dirs = os.listdir()
dirs = 'MM.csv'

Data = (pd.read_csv(dirs))

os.chdir(owd)
Train = Data.Close.values
scaler = MinMaxScaler(feature_range=(0, 1))
Train = scaler.fit_transform(Train[:, np.newaxis])

# split into train and test sets
train_size = int(len(Train) * 0.67)
test_size = len(Train) - train_size
train, test = Train[0:train_size,:], Train[train_size:len(Train),:]

plt.plot(train)
plt.plot(test)
plt.show()

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=1)

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
plt.show()

model.save('nifty_fifty.hdf5')