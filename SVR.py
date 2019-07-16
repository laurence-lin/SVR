import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

from sklearn.svm import SVR 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import TimeSeriesPipeline as timeseries
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
import tensorflow as tf

'''
Energy consume regression by SVR
'''

data = pd.read_csv('energydata.csv')
print(data.head())
print(data.values.shape)
y = data['Appliances']
print(data.columns)
x = data.iloc[:, 2:-2] # don't consider random variables & date

# Data preprocessing
# Scaling
x = np.array(x)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y.values.reshape(-1, 1))

# Convert to time series pipeline
time_step = 7
x_data, y_data = timeseries.TimeSeriesData(x, y, time_step)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2)

print('x_train:', x_train.shape)
print('y_train:', y_train.shape)

# Run model training
svr = SVR(C = 100, 
          kernel = 'rbf',
          gamma = 'scale')
'''
If gamma = 'auto', gamma = 1/n_features
'''
svr.fit(x_train, y_train)
score = svr.score(x_train, y_train)
print(score)


'''Show performance in figure'''
# Inverse scaling for checking performance
predict = svr.predict(x_test).reshape(-1, 1)
predict = scaler.inverse_transform(predict)
y_test = scaler.inverse_transform(y_test)

# error performance: RMSE
rmse = np.mean( np.sqrt( (predict - y_test)**2 ) )

# Show performance
fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
ax.plot(y_test, color = 'red')
ax.title.set_text('Original')


ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(predict, color = 'blue')
ax2.title.set_text('SVR')

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(y_test, color = 'red', label = 'Original')
ax3.plot(predict, color = 'blue', label = 'SVR')
ax3.legend()
ax3.title.set_text('Comparirson  RMSE: %s'%rmse)

plt.show()












