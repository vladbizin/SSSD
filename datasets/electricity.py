import os
import pickle
from copy import deepcopy
import numpy as np
import tsdb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split




# ELECTRICITY LOAD DIAGRAMS

data = tsdb.load('electricity_load_diagrams')

X = data['X'].values

# cut off tails to split into samples of size 100
X = X[:100 * (X.shape[0]//100)]

# split into samples of size 100
X = np.array(np.split(X, X.shape[0]//100))

# randomly split with ratio as in papers
# test          : val       : train
# 10 months     : 10 months : 18 months
# 369           : 369       :664
X_train, tmp = train_test_split(X, train_size=18/38, shuffle=True)
X_test, X_val = train_test_split(tmp, train_size=0.5, shuffle=True)


# impute X_val with missing values - 10% missing
X_val_ori = X_val.copy()
mask = np.random.uniform(size = X_val.shape) < 0.5
X_val = np.where(mask, X_val, np.nan)

test_scalers = []
val_scalers = []
train_scalers = []

# scale data to (-1, 1) value range

for i in range(X_test.shape[0]):
    scaler = MinMaxScaler((-1,1))
    X_test[i] = scaler.fit_transform(X_test[i])
    test_scalers.append(deepcopy(scaler))

for i in range(X_val.shape[0]):
    scaler = MinMaxScaler((-1,1))
    X_val[i] = scaler.fit_transform(X_val[i])
    val_scalers.append(deepcopy(scaler))

for i in range(X_train.shape[0]):
    scaler = MinMaxScaler((-1,1))
    X_train[i] = scaler.fit_transform(X_train[i])
    train_scalers.append(deepcopy(scaler))


dirname = os.path.dirname(__file__)
dirname = os.path.join(dirname, 'datasets/electricity/')
if not os.path.isdir(dirname):
        os.makedirs(dirname)
        os.chmod(dirname, 0o775)


with open(dirname + 'test_scalers.pkl','wb') as f:
    pickle.dump(test_scalers, f)
with open(dirname + 'val_scalers.pkl','wb') as f:
    pickle.dump(val_scalers, f)
with open(dirname + 'train_scalers.pkl','wb') as f:
    pickle.dump(train_scalers, f)

np.save(dirname + 'val.npy', X_val)
np.save(dirname + 'val_ori.npy', X_val_ori)
np.save(dirname + 'test.npy', X_test)
np.save(dirname + 'train.npy', X_train)