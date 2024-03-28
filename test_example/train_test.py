from SSSD_module.model import SSSDS4
import numpy as np
import os

# make sure to place this file and 
# 'test_data' folder in same directory

dirname = os.path.join(
    os.path.dirname(__file__),
    'test_data/datasets/ETTm'
)

train = np.load(dirname + '/train.py')
val = np.load(dirname + '/val.py')
val_ori = np.load(dirname + '/val.py')


train_config={
    "output_directory": "/results",
    "epochs": 100,
    "epochs_per_ckpt": 50,
    "learning_rate": 2e-4,
    "only_generate_missing": True,
    "missing_r": "rand",
    "batch_size": 150,
    "verbose": True,
}

model =  SSSDS4(*train.shape[1:])
model.set_train_config(**train_config)

model.train(train, (val, val_ori))