import numpy as np
from sssd.model import SSSDS4

# read data needed for training and validation
# these .npy files can be obtained via data preparation
# see README.md fot more info
train = np.load("your_path_to_train_file.npy")
val = np.load("your_path_to_val_file.npy")
val_ori = np.load("your_path_to_val_ori_file.npy")


# initialize training parameters
# see config setting in .md file
# see README.md fot more info
train_config={
    "output_directory": "your_directory",
    "epochs": 1000,
    "epochs_per_ckpt": 250,
    "epochs_per_val": 25,
    "learning_rate": 1e-3,
    "only_generate_missing": True,
    "missing_r": "rand",
    "batch_size": train.shape[0]//10,
    "verbose": False
}

# initialize model
model = SSSDS4(train.shape[1], train.shape[2])
model.set_train_config(**train_config)

# train the model
model.train(train, (val, val_ori))