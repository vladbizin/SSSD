import os
import pickle
from copy import deepcopy
import numpy as np
import tsdb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ELECTRICITY TRANSFORMER TEMPERATURE

def sliding_window_generator(X, window_size, stride):
    out = []
    for i in range(0, X.shape[0] - window_size, stride):
        out.append(X[i: i + window_size])
    return np.array(out)


data = tsdb.load('electricity_transformer_temperature')

X = data['ETTm1'].values

# randomly split with ratio as in papers
# test      : val       : train
# 4 months  : 4 months  : 16 months
X_train, tmp = train_test_split(X, train_size=16/24, shuffle=True)
X_test, X_val = train_test_split(tmp, train_size=0.5, shuffle=True)

# cut off tails to split into samples of size 24
X_test = X_test[:24 * (X_test.shape[0]//24)]          # (11592, 7)
X_val = X_val[:24 * (X_val.shape[0]//24)]             # (11592, 7)
X_train = X_train[:24 * (X_train.shape[0]//24)]       # (46440, 7)

# split into samples of size 24
X_test = np.array(np.split(X_test, X_test.shape[0]//24))    # (483, 24, 7)
X_val = np.array(np.split(X_val, X_val.shape[0]//24))       # (483, 24, 7)

# generate samples with window size 24 and stride 12
X_train = sliding_window_generator(X_train, 24, 12)         # (3868, 24, 7)

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


# impute X_val with missing values - 10% missing
X_val_ori = X_val.copy()
mask = np.random.uniform(size = X_val.shape) < 0.9
X_val = np.where(mask, X_val, np.nan)

dirname = os.path.dirname(__file__)
dirname = os.path.join(dirname, 'datasets/ETTm/')
if not os.path.isdir(dirname):
        os.makedirs(dirname)
        os.chmod(dirname, 0o775)

# save scaler for unscaling
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