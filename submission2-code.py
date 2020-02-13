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
train_data = df.values
t = train_data[:,0]
X = train_data[:,1:27]
y = train_data[:,27].astype(int)
N = y.shape[0]

"""

from sklearn.neural_network import MLPRegressor

clf = MLPRegressor(max_iter=100, shuffle = False)
clf.fit(X, y)

"""

from sklearn.ensemble import RandomForestRegressor as Jesus

num_folds = 5
kf = KFold(n_splits=num_folds)
    
for train_index, val_index in kf.split(X[0:N]):

    train_data_x, val_data_x = X[train_index], X[val_index]
    train_data_y, val_data_y = y[train_index], y[val_index]

    baby_jesus = Jesus(n_estimators=100, min_samples_leaf=10, n_jobs=-1)
    baby_jesus.fit(train_data_x, train_data_y)
    print(baby_jesus.score(val_data_x, val_data_y))



"""
baby_jesus = Jesus(n_estimators=100, n_jobs=-1)
baby_jesus.fit(X, y)

df = pd.read_csv("test.csv")
df = df.fillna(0)
test_data = df.values
ids = test_data[:,0]
X_test = test_data[:,1:]

output = baby_jesus.predict(X_test)
output = np.clip(output, 0, 1)

print(baby_jesus.score(X, y))

f = open("submission2.csv", "wt")
f.write("id,Predicted\n")

for ID, prob in zip(ids, output):
    f.write(f"{int(ID)},{prob}\n")

f.close()
"""