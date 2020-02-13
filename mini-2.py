import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as torch_data
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.linear_model import LinearRegression

df = pd.read_csv("train.csv")
df = df.fillna(0)
# df = df.select_dtypes(include=[np.number]).apply(stats.zscore)
train_data = df.values
t = train_data[:,0]
X = train_data[:,1:27]
y = train_data[:,27].astype(int)
N = y.shape[0]


"""
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))
print(reg.predict(X))
"""

from sklearn.linear_model import Ridge
clf = Ridge(alpha=100.0)
clf.fit(X, y)

df = pd.read_csv("test.csv")
df = df.fillna(0)
# df = df.select_dtypes(include=[np.number]).apply(stats.zscore)
train_data = df.values
ids = train_data[:,0]
X_test = train_data[:,1:]
N = y.shape[0]

from sklearn import preprocessing


output = clf.predict(X_test)
output = np.clip(output, 0, 1)
# min_max_scaler = preprocessing.MinMaxScaler()
# output = min_max_scaler.fit_transform(output.reshape((-1, 1)))

f = open("submission1.csv", "wt")
f.write("id,Predicted\n")

for ID, prob in zip(ids, output):
    f.write(f"{int(ID)},{prob}\n")

f.close()



"""
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

rng = np.random.RandomState(1)

regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6),
                          n_estimators=300, random_state=rng)

regr.fit(X, y)
print(regr.score(X, y))
print(regr.predict(X))
"""