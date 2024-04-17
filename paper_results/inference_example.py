import pickle
import numpy as np

from sssd.model import SSSDS4
from sssd.utils.util import mar, bom, rbm, tsf


# load test data and test scalers
with open("your_path_to_scalers.pkl", 'rb') as f:
    test_scalers = pickle.load(f)
test = np.load("your_path_to_test_file.npy")

# initalize model and load model state
model = SSSDS4()
model.load_state("your_path_to_output_directiry")


# dictionary with results
res = {
    "mar": {},
    "bom": {},
    "rbm": {},
    "tsf": {},
    "original": None
}

# masking functions dictionary
funcs = {
    "mar": mar,
    "bom": bom,
    "rbm": rbm,
    "tsf": tsf
}

# inverse scaling testing data
original = np.zeros_like(test)
for i in range(test.shape[0]):
    original[i] = (test_scalers[i]).inverse_transform(test[i])
res["original"] = original

# how many times we impute missing data
N = 5

# this function saves imputation results to our dictionary
def impute(mode, mr):
    res[mode][mr] = {}
    missing_mask = funcs[mode](test, mr)
    res[mode][mr]["missing_mask"] = missing_mask
    res[mode][mr]["imputation"] = []

    for _ in range(N):
        test_c = test.copy()
        test_c[missing_mask] = np.nan
        imputation = model.predict(test_c, test.shape[0])

        for i in range(imputation.shape[0]):
            imputation[i] = (test_scalers[i]).inverse_transform(imputation[i])

        res[mode][mr]["imputation"].append(imputation)

    res[mode][mr]["imputation"] = np.array(res[mode][mr]["imputation"])


for mr in np.linspace(0.1, 0.9, 9):
    mr = round(mr, 1)
    print(mr)
    impute("mar", mr)
    impute("bom", mr)
    impute("rbm", mr)
    impute("tsf", mr)


with open("your_path_to_imputation_result.pkl", "wb") as output_file:
    pickle.dump(res, output_file)