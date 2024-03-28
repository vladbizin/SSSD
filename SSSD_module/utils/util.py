import os
import numpy as np
import torch
from torch.utils.data import Dataset

def get_mask_rbm(obs_mask, ratio):
    cond_mask = obs_mask

    L = obs_mask.shape[1]
    num_masked = round(L * ratio)
    l = np.random.choice(
        np.array(range(L))[-num_masked],
        obs_mask.shape[0]
    )
    r = l + num_masked

    for i in range(obs_mask.shape[0]):
        cond_mask[l[i]:r[i]] = 0
    return cond_mask


def get_mask_bm(obs_mask, ratio):
    cond_mask = obs_mask

    L = obs_mask.shape[1]
    num_masked = round(L * ratio)
    l = np.random.choice(
        np.array(range(L))[-num_masked]
    )
    r = l + num_masked

    cond_mask[l:r] = 0
    return cond_mask


def get_mask_rm(obs_mask, missing_rate):
    num_observed = obs_mask.sum().item()
    num_masked = round(num_observed * missing_rate)
    
    masked = np.random.choice(
        np.arange(torch.numel(obs_mask)).reshape(obs_mask.shape)[obs_mask], 
        num_masked, replace=False
    )
    cond_mask = obs_mask.reshape(-1)
    cond_mask[:] = False
    cond_mask[masked] = True
    cond_mask = torch.reshape(cond_mask, obs_mask.shape)
    return cond_mask

def get_mask_tf(obs_mask, ratio):
    cond_mask = obs_mask
    L = obs_mask.shape[1]
    num_masked = round(L * ratio)
    cond_mask[-num_masked:] = 0
    return cond_mask


class TS_Dataset(Dataset):

    def __init__(
        self,
        data,
        obs_mask,
        dataset_mode,
        missing_mode = "rm",
        missing_r = "rand",
    ):
        super().__init__()
        if dataset_mode == "Validation":
            self.data = torch.from_numpy(data[0])
            self.data_ori = torch.from_numpy(data[1])
            self.cond_mask = torch.from_numpy(obs_mask[0])
            self.obs_mask = torch.from_numpy(obs_mask[1])
        else:
            self.data = torch.from_numpy(data)
            self.obs_mask = torch.from_numpy(obs_mask)
        self.dataset_mode = dataset_mode
        self.missing_mode = missing_mode
        self.missing_r = missing_r

    def __len__(self) -> int:
        return self.data.shape[0]
    
    def get_mask(self, obs_mask) -> torch.Tensor:

        # missing ratio
        if self.missing_r=="rand":
            ratio = np.random.rand()  
        else:
            ratio = self.missing_r

        if self.missing_mode == "rm":
            cond_mask = get_mask_rm(obs_mask, ratio)
        elif self.missing_mode == "bm":
            cond_mask = get_mask_bm(obs_mask, ratio)
        elif self.missing_mode == "rbm":
            cond_mask = get_mask_rbm(obs_mask, ratio)
        elif self.missing_mode == "tf":
            cond_mask = get_mask_tf(obs_mask, ratio)

        return cond_mask

    def __getitem__(self, idx: int):
        if self.dataset_mode == "Validation":
            ts = self.data[idx].float()
            ts_ori = self.data_ori[idx].float()
            obs_mask = self.obs_mask[idx].float()
            cond_mask = self.cond_mask[idx].float()
            return (ts, ts_ori), obs_mask, cond_mask
        
        ts = self.data[idx].float()
        obs_mask = self.obs_mask[idx].float()
        if self.dataset_mode == "Training":
            cond_mask = self.get_mask(self.obs_mask[idx]).float()
            return ts, obs_mask, cond_mask
        elif self.dataset_mode == "Inference":
            return ts, obs_mask



def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]                    # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])                           # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
                                                            # / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def sampling(net, diffusion_hyperparams, cond, mask):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    size = cond.size()

    x = std_normal(size)

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            x = x * (1 - mask).float() + cond * mask.float()

            # use the corresponding reverse step
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  

            # predict \epsilon according to \epsilon_\theta
            epsilon_theta = net((x, cond, mask, diffusion_steps,))

            # update x_{t-1} to \mu_\theta(x_t)
            x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            
            # add the variance term to x_{t-1}
            if t > 0:
                x = x + Sigma[t] * std_normal(size)

    return x


def training_loss(net, loss_fn, X, diffusion_hyperparams):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors       
    
    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, K, L = audio.shape  # B is batchsize, K is number of features, L is audio length

    # randomly sample diffusion steps from 1~T
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()  

    # N(0,1)
    z = std_normal(audio.shape)
    z = audio * mask.float() + z * (1 - mask).float()

    # compute x_t from q(x_t|x_0)
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
        1 - Alpha_bar[diffusion_steps]) * z  
    
    # predict \epsilon according to \epsilon_\theta
    epsilon_theta = net(
        (transformed_X, cond, mask, diffusion_steps.view(B, 1),)
    )  

    return loss_fn(epsilon_theta[loss_mask], z[loss_mask])