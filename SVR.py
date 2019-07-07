import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import SVR 
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('wpbc.data')
print(data.columns)
print(data.info())
data = np.array(data)

x = data[:, 3].reshape(len(data[:, 3]), 1)
y = data[:, 5].reshape(len(data[:, 5]), 1)
normalize = StandardScaler()
normalize.fit(x)
#normalize.transform(x)
print(x)

normalize.fit(y)
#normalize.transform(y)
svr = SVR(kernel = 'linear', C = 3)
svr.fit(x, y)
score = svr.score(x, y)
print('Score', score)
predict = svr.predict(x)

plt.scatter(data[:, 3], data[:, 5], color = 'blue')
plt.plot(data[:, 3], predict, 'r')
plt.xlabel('perimeter')
plt.ylabel('radius')
plt.title('Breast cancer')
plt.show()