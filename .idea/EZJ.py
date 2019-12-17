from datetime import *
import pandas_datareader.data as data
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

start = datetime(2019, 6, 1)
end = datetime.now()
EZJ = data.DataReader('EZJ.L', 'yahoo', start, end)

df = EZJ.Close

order = 0
for i in range(order):
    for i in reversed(range(len(df))):
        df[i] -= df[i-1]
    df = df[1:]


plt.plot(df)
plt.show()

result = adfuller(df.values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])



plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.values); axes[0, 0].set_title('Original Series')
plot_acf(df.values, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(np.diff(df.values)); axes[1, 0].set_title('1st Order Differencing')
plot_acf(np.diff(df.values), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(np.diff(np.diff(df.values))); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(np.diff(np.diff(df.values)), ax=axes[2, 1])

plt.show()

# 1,1,2 ARIMA Model
model = ARIMA(df.values, order=(0,1,2))
model = model.fit(disp=10)
print(model.summary())

model.plot_predict(dynamic=False)
plt.show()


