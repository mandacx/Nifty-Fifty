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
from tensorflow.keras.layers import LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



def split_sequences(share, n_steps, target, look_ahead=7, start=datetime(2005, 6, 1), end=datetime.now(), ):
	dataset = data.DataReader(share, 'yahoo', start, end)
	cols = dataset.columns

	index = dataset.index[n_steps+look_ahead-1:]
	dataset = dataset.values
	maxes = np.max(dataset, axis=0)
	mins = np.min(dataset, axis=0)
	ranges = maxes - mins
	dataset = (dataset - mins) / ranges
	#scaler.fit_transform(dataset)
	dataset = pd.DataFrame(dataset, columns = cols)

	X, y = list(), list()
	for i in range(len(dataset)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix+look_ahead-1 > len(dataset)-1:
			break
		# gather input and output parts of the pattern

		seq_x = dataset[i:end_ix].values
		seq_y = dataset[end_ix+look_ahead-1: end_ix+look_ahead][target].values

		X.append(seq_x)
		y.append(seq_y)
	X=np.asarray(X)
	y=np.asarray(y)

	return X, y, ranges, mins, index

def plot_history(history):
    keys = history.history.keys()
    # eprint(keys)
    for key in filter(lambda k:"val_" not in k,  keys):
        plt.plot(history.history[key])
        plt.plot(history.history['val_'+key ])
        plt.title( key)
        plt.ylabel('key')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

def inverse_scale(x, range, min, index=2):
	return x*range[index]+min[index]

def get_data(share):
	x, y, ranges, mins, index = split_sequences(share = share, n_steps=32, target='Open')

	train_split = 0.9
	train_split = int(x.shape[0]*train_split)
	x_train, y_train = x[:train_split], y[:train_split]
	x_test, y_test = x[train_split:], y[train_split:]
	return x_train, y_train, x_test, y_test, ranges, mins, index

x_train, y_train, x_test, y_test, ranges, mins, index = get_data('PMO.L')
# define model
model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=x_train[1].shape[0:2]))
model.add(Dropout(0.5))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(np.squeeze(y_train[1].shape)))
model.compile(optimizer='adam', loss='mse')



companies = ['PMO.L', 'RDSB.L', 'EZJ.L']
companies = ['EZJ.L']
for share in companies:
	model.load_weights('LSTM.hdf5')

	x_train, y_train, x_test, y_test, ranges, mins, index = get_data(share)
	model.fit(x_train, y_train,
			  batch_size=32,
			  epochs=10,
			  verbose=1)
model.save_weights('LSTM.hdf5')

y_scaled_train = np.zeros([y_train.shape[0]+y_test.shape[0], 1])
y_scaled_train[:] = np.nan
y_scaled_train[:y_train.shape[0]] = inverse_scale(y_train, ranges, mins)
y_scaled_train[-y_test.shape[0]:] = inverse_scale(y_test, ranges, mins)

p_scaled_train = np.zeros(y_scaled_train.shape)
p_scaled_train[:] = np.nan
p_scaled_train[:y_train.shape[0]] = inverse_scale(model.predict(x_train), ranges, mins)

p_scaled_test = np.zeros(y_scaled_train.shape)
p_scaled_test[:] = np.nan
p_scaled_test[-y_test.shape[0]:] = inverse_scale(model.predict(x_test), ranges, mins)


plt.plot(index, y_scaled_train)
plt.plot(index, p_scaled_train)
plt.plot(index, p_scaled_test)
plt.show()

plot_history(model.history)


# error = y_test-y_pred
# mean = np.mean(np.abs(error))
# stddev = np.std(error)
#
# print(mean,stddev)

# y_train = inverse_scale(y_train, ranges, mins)
# y_pred =
# error = y_train-y_pred
