import os
import pickle
from copy import deepcopy
import numpy as np
import tsdb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def mar(X, missing_rate):
    obs_mask = ~np.isnan(X)
    num_observed = obs_mask.sum()
    num_masked = round(num_observed * missing_rate)
    
    masked = np.random.choice(np.arange(obs_mask.size).reshape(obs_mask.shape)[obs_mask], num_masked, replace=False)

    missing_mask = obs_mask.reshape(-1)
    missing_mask[:] = False
    missing_mask[masked] = True
    missing_mask = missing_mask.reshape(obs_mask.shape)

    return missing_mask

# PhysioNet 2012

data = tsdb.load('physionet_2012')

X = data['X'].drop(columns = data['static_features'])
all_data = np.empty((11988, 49, 37))
all_data[:] = np.nan
for i, id in enumerate(X['RecordID'].unique()):
    tmp = X.loc[X['RecordID']==id]
    time = tmp['Time'].values
    tmp = tmp.drop(columns=['Time', 'RecordID'])
    all_data[i, time] = tmp


# randomly split with ratio as in papers
X_train, X_test = train_test_split(all_data, train_size=0.8, shuffle=True)
X_train, X_val = train_test_split(X_train, train_size=0.8, shuffle=True)


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
mask = ~mar(X_val, 0.1)
X_val = np.where(mask, X_val, np.nan)


dirname = os.path.dirname(__file__)
dirname = os.path.join(dirname, 'datasets/physionet/')
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