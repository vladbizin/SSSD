from SSSD_module.model import SSSDS4
import numpy as np
import pickle
import os

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


# make sure to place this file and 
# 'test_data' folder in same directory

dirname = os.path.join(
    os.path.dirname(__file__),
    'test_data/datasets/ETTm'
)

test = np.load(dirname + 'test.npy')
with open(dirname + 'test_scalers.pkl', 'rb') as f:
    test_scalers = pickle.load(f)

model =  SSSDS4(*test.shape[1:])
model.load_state('results')


original = np.zeros_like(test)
for i in range(test.shape[0]):
    original[i] = (test_scalers[i]).inverse_transform(test[i])


missing_mask = mar(test, 0.1)
test_c = test.copy()
test_c[missing_mask] = np.nan


imputation = model.predict(test_c, test_c.shape[0])


mae_error = np.abs(imputation[missing_mask] - test_c[missing_mask]).mean()
print(
    f'mae error: {mae_error.round(2)}'
)

with open(dirname+'prediction.pkl', "wb") as output_file:
    pickle.dump(imputation, output_file)