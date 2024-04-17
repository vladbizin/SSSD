import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from gluonts.dataset.repository.datasets import  get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from copy import deepcopy
import os
import pickle


def sliding_window_generator(X, window_size, stride):
    out = []
    for i in range(0, X.shape[0] - window_size, stride):
        out.append(X[i: i + window_size])
    return np.array(out)


# EXCHANGE

data = get_dataset("exchange_rate", regenerate=False)

train_grouper = MultivariateGrouper(
    max_target_dim=int(data.metadata.feat_static_cat[0].cardinality)
    )

test_grouper = MultivariateGrouper(
    num_test_dates=int(len(data.test) / len(data.train)),
    max_target_dim=int(data.metadata.feat_static_cat[0].cardinality)
    )


# L = 60
L = 2 * data.metadata.prediction_length

X_train = train_grouper(data.train)[0]['target'].T
X_test = test_grouper(data.test)[0]['target'].T


# split train into train and validation 1:9 ratio
X_train, X_val = train_test_split(X_train, train_size=0.9, shuffle=True, random_state=42)

# generate samples with window size 60 and stride 10
X_train = sliding_window_generator(X_train, L, L//4)    #(361, 60, 8)

# cut off tails to split into samples of size 60
X_val = X_val[:L*(X_val.shape[0]//L)]
X_test = X_test[:L*(X_test.shape[0]//L)]

# split into samples of size 60
X_val = np.array(np.split(X_val, X_val.shape[0]//L))    #(10, 60, 8)
X_test = np.array(np.split(X_test, X_test.shape[0]//L)) #(101, 60, 8)


# scale data to (-1, 1) value range

test_scalers = []
val_scalers = []
train_scalers = []

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


X_val_ori = X_val.copy()
X_val[:, L//2:, :] = np.nan


dirname = os.path.dirname(__file__)
dirname = os.path.join(dirname, 'datasets/exchange/')
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