# SSSD

Structured Space State Diffusion Model for Time Series Imputation

## Quick Start

### Installation
To isntall the SSSD library with pip you can simply use

```bash
pip install git+https://github.com/vladbizin/SSSD
```

If you want to additionally create a virtual environment first, run
```bash
python -m venv your_environment_name
```
then run
```bash
your_environment_name\Scripts\activate.bat
```
And when finished run
```bash
deactivate
```

### Usage

#### Training
To train a model:
1. Prepare your datasets:
* make sure all training, validating and testing datasets are of shape `number of samples, time series length, number of features (channels)`
* preferably normalize your dataset (results in better preformance)
* if you want to have validating datasets, make sure you have two of them, the original one and the one with missing values
* make sure all missing values are of type `np.nan`
2. Prepare your config dictionaries
3. Run the model

```python
from sssd.model import SSSD
import numpy as np

train = np.load("path_to_training_data.npy")
val_ori = np.load("path_to_validating_data_without_missing_values.npy")
val = np.load("path_to_validating_data_with_missing_values.npy")

train_config={
    "output_directory": "your_directory",
    "epochs": 1000,
    "epochs_per_ckpt": 100,
    "epochs_per_val": 10,
    "learning_rate": 1e-3,
    "only_generate_missing": True,
    "missing_r": "rand",
    "batch_size": train.shape[0]//10
}

# initialize model
model = SSSDS4(train.shape[1], train.shape[2])
model.set_train_config(**train_config)

# train the model
model.train(train, (val, val_ori))
```
If you don't wand to use validating dataset then just pass the trainin one.  
You can see a more thourughful example in the `Results Reproduction` section.

#### Inference
To impute missing results:
1. Prepare the testing dataset: make sure the missing values are of type `np.nan`
2. Load your model (the path should contain a `.pkl` file with saved state, preferrably,
it should be the same directory you passed to `ouput_directiry` in `train_config` of the training stage)
3. Impute missing values

```python
from sssd.model import SSSD
import numpy as np

test = np.load("your_path_to_testing_data.npy")

model = SSSDS4()
model.load_state("yout_directory")

# second parameter is batch_size during inference
imputation = model.predict(test, test.shape[0])
```
You can see a more thourughful example in the `Results Reproduction` section.

## Results Reproduction

### Datasets preparation
All dataset handling files are in the `datasets` folder of the project.

### Training
Sample train file `paper_results/train_example.py` provides an example on how to train the model with given parameters on given dataset.  
To run it successfully make sure to:
1. Preprocess training, testing, and validating datasets to the form needed. (you can simply run preprocessing scripts from `"datasets"` folder)
2. Input paths to training data
3. Input saving path (the `output_directiry` parameter)

#### Training parameters
In the table below you will find training parameters values used for each dataset.  
`only_generate_missing` was set to `True`, `missing_r` was set to `"rand"` for all datasets.

|          Dataset         | Epochs | learning_rate |     batch_size     |
|--------------------------|--------|---------------|--------------------|
| physionet                | 1000   | 1e-3          | 256                |
| ettm                     | 1000   | 1e-3          | 500                |
| electricity              | 500    | 2e-4          | 20                 |
| air quality              | 2000   | 1e-3          | 64                 |
| solar                    | 1000   | 1e-3          | train.shape[0]//10 |
| exchange                 | 1000   | 1e-3          | train.shape[0]     |
| electricity (prediction) | 1000   | 2e-4          | train.shape[0]//10 |

###  Inference
Sample inference file `paper_results/inference_example.py` provides an example on how to use the trained model for missing data imputation.
To run it successfully make sure to:
1. Input paths to load `test` and `test_scalers`
2. Input path to save your results
3. Set your number of imputations (`N` parameter)

In our work we set it to 5 for every dataset.

### Metrics calculation and plots
You can find these in `paper_results/metrics_example.ipynb`
  
If you are missing some information, you may find it in the paper.

## Documentation

docstrings and other documentation will be added

## Acknowledgments
We thank the following repositories, papers and their authors and do not claim any authorship on their achievments.

* [Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models](https://arxiv.org/abs/2208.09399), [GitHub repo](https://github.com/AI4HealthUOL/SSSD)
* [PyPOTS](https://github.com/WenjieDu/PyPOTS)

## License
[MIT]() License
