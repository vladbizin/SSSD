import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional




# (n_samples, n_time_steps, n_features)

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

def bom(X, missing_rate):
    # for datasets without missing values
    w = round(missing_rate * X.shape[1])
    l = np.random.choice(
        np.arange(X.shape[1])[:-w]
    )
    r = l + w
    missing_mask = np.zeros_like(X).astype(bool)
    missing_mask[:, l:r, ] = True
    return missing_mask

def rbm(X, missing_rate):
    w = round(missing_rate * X.shape[1])
    l = np.random.choice(
        np.arange(X.shape[1])[:-w],
        X.shape[2]
    )
    r = l + w
    missing_mask = np.zeros_like(X).astype(bool)
    for i in range(X.shape[2]):
        missing_mask[:, l[i]: r[i], i] = True
    return missing_mask

def tsf(X, missing_rate):
    w = round(missing_rate * X.shape[1])
    missing_mask = np.zeros_like(X).astype(bool)
    missing_mask[:, -w:, :] = True
    return missing_mask


def get_line_plot_mask(arr):
    line_plot_mask = np.zeros_like(arr).astype(bool)
    line_plot_mask[:-1] = ~np.isnan(arr[:-1]) & ~np.isnan(arr[1:])
    line_plot_mask[1:] |= ~np.isnan(arr[:-1]) & ~np.isnan(arr[1:])
    return line_plot_mask

def get_segments(mask):
    res = []
    l = 0
    while l < mask.shape[0] - 2:
        if not mask[l]:
            l += 1
            continue
        r = l + 1
        while mask[r] and r < mask.shape[0] - 1:
            r += 1
        ins_mask = np.zeros_like(mask).astype(bool)
        ins_mask[l:r] = True
        res.append(ins_mask)
        l = r + 1
    return np.array(res)

def get_add_mask(obs_mask, imp_mask):
    add_mask = np.zeros_like(obs_mask).astype(bool)
    add_mask[:-1] = obs_mask[:-1] & imp_mask[1:]
    add_mask[1:] |= obs_mask[1:] & imp_mask[:-1]
    return add_mask

def plot_data(
    X: np.ndarray,
    X_ori: np.ndarray,
    X_imputed: np.ndarray,
    mode: str,
    sample_idx: Optional[int] = None,
    n_rows: int = 10,
    n_cols: int = 4,
    fig_size: Optional[list] = None,
):
    """Plot the imputed values, the observed values, and the evaluated values of one multivariate timeseries.
    The observed values are marked with red 'x',  the evaluated values are marked with blue 'o',
    and the imputed values are marked with solid green line.

    Parameters
    ----------
    X : ndarray,
        The observed values

    X_ori : ndarray,
        The evaluated values

    X_imputed : ndarray,
        The imputed values

    sample_idx : int,
        The index of the sample to be plotted.
        If None, a randomly-selected sample will be plotted for visualization.

    n_rows : int,
        The number of rows in the plot

    n_cols : int,
        The number of columns in the plot

    fig_size : list,
        The size of the figure
    """

    vals_shape = X.shape
    assert (
        len(vals_shape) == 3
    ), "vals_obs should be a 3D array of shape (n_samples, n_steps, n_features)"
    n_samples, n_steps, n_features = vals_shape

    if sample_idx is None:
        sample_idx = np.random.randint(low=0, high=n_samples)
        
    if fig_size is None:
        fig_size = [24, 36]

    n_k = n_rows * n_cols
    K = np.min([n_features, n_k])
    L = n_steps
    plt.rcParams["font.size"] = 16
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(fig_size[0], fig_size[1])
    )

    X_imputed_mean = np.mean(X_imputed, axis = 0)
    X_impued_std = np.std(X_imputed, axis = 0)
    for k in range(K):
        df = pd.DataFrame({"x": np.arange(0, L), "m": X_imputed_mean[sample_idx, :, k], "s": X_impued_std[sample_idx, :, k]})
        df1 = pd.DataFrame({"x": np.arange(0, L), "val": X[sample_idx, :, k]})
        df2 = pd.DataFrame({"x": np.arange(0, L), "val": X_ori[sample_idx, :, k]})
        row = k // n_cols
        col = k % n_cols
    
        # observed
        obs_line_plot_mask = get_line_plot_mask(df1["val"].values)
        axes[row][col].plot(df1.x, np.where(obs_line_plot_mask, df1.val, np.nan), c="blue", linestyle="solid")    
        axes[row][col].plot(df1.x, np.where(~obs_line_plot_mask, df1.val, np.nan), "o", c="blue")

        # # evaluated
        # axes[row][col].plot(df2.x, np.where(add_mask, df1.val, df2.val), c="red", linestyle="solid")

        # # imputed
        # axes[row][col].plot(df.x, np.where(add_mask, df1.val, df.val), c="green", linestyle="solid")

        obs_mask = ~np.isnan(df1.val.values)
        imp_mask = ~np.isnan(df.m.values)
        for imp_segment in get_segments(imp_mask):
            add_mask = get_add_mask(obs_mask, imp_segment)
            # evaluated
            axes[row][col].plot(df2.x, np.where(add_mask, df1.val, df2.val), c="red", linestyle="solid")
            # imputed
            axes[row][col].plot(df.x, np.where(add_mask, df1.val, df.m), c="green", linestyle="solid")
            axes[row][col].fill_between(df.x, np.where(add_mask, df1.val, df.m - 1 * df.s), np.where(add_mask, df1.val, df.m + 1 * df.s), color="green", alpha=.4)
            axes[row][col].fill_between(df.x, np.where(add_mask, df1.val, df.m - 2 * df.s), np.where(add_mask, df1.val, df.m + 2 * df.s), color="green", alpha=.4)
            axes[row][col].fill_between(df.x, np.where(add_mask, df1.val, df.m - 3 * df.s), np.where(add_mask, df1.val, df.m + 3 * df.s), color="green", alpha=.4)

        if col == 0:
            plt.setp(axes[row, 0], ylabel="value")
        if row == -1:
            plt.setp(axes[-1, col], xlabel="time")



        # # observed
        # obs_line_plot_mask = get_line_plot_mask(df1["val"].values)
        # axes[row][col].plot(df1.x, np.where(obs_line_plot_mask, df1.val, np.nan), c="blue", linestyle="solid")    
        # axes[row][col].plot(df1.x, np.where(~obs_line_plot_mask, df1.val, np.nan), "o", c="blue")

        # # evaluated
        # eval_line_plot_mask = get_line_plot_mask(df2["val"].values)
        # axes[row][col].plot(df2.x, np.where(eval_line_plot_mask, df2.val, np.nan), c="red", linestyle="solid")    
        # axes[row][col].plot(df2.x, np.where(~eval_line_plot_mask, df2.val, np.nan), "o", c="red")

        # # imputed
        # imp_line_plot_mask = get_line_plot_mask(df["val"].values)
        # axes[row][col].plot(df.x, np.where(imp_line_plot_mask, df.val, np.nan), c="green", linestyle="solid")    
        # axes[row][col].plot(df.x, np.where(~imp_line_plot_mask, df.val, np.nan), "o", c="green")

        if col == 0:
            plt.setp(axes[row, 0], ylabel="value")
        if row == -1:
            plt.setp(axes[-1, col], xlabel="time")


        
        # axes[row][col].plot(df1.x, df1.val, c="blue", linestyle="solid")    # observed
        # if mode == 'mar':
        #     axes[row][col].plot(df2.x, df2.val, "o", c="red")     # evaluated
        #     axes[row][col].plot(df.x, df.val, "o", c="green")     # imputed
        # else:
        #     axes[row][col].plot(df2.x, df2.val, c="red", linestyle="solid")     # evaluated
        #     axes[row][col].plot(df.x, df.val, c="green", linestyle="solid")     # imputed
        # if col == 0:
        #     plt.setp(axes[row, 0], ylabel="value")
        # if row == -1:
        #     plt.setp(axes[-1, col], xlabel="time")
    plt.show()


def nlpd_metric(y_true, m, std):
    return np.mean((m - y_true)**2 / (2 * std**2) + np.log(std) + 0.5 * np.log(2 * np.pi))


def picp_metic(y_true, y_pred, bound):
    satisfies_upper_bound = y_true <= y_pred + bound
    satisfies_lower_bound = y_true >= y_pred - bound
    return np.mean(satisfies_upper_bound * satisfies_lower_bound)
