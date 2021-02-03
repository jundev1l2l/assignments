import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from random import sample


"""
data generating tool
"""
class NormalizerClass(object):
    """
    Normalizer Class
    """
    def __init__(self,raw_data,eps=1e-8):
        self.raw_data = raw_data
        self.eps      = eps
        self.mu       = torch.mean(self.raw_data,dim=0)
        self.std      = torch.std(self.raw_data,dim=0)
        self.nzd_data = self.get_nzdval(self.raw_data)
        self.org_data = self.get_orgval(self.nzd_data)
        self.max_err  = torch.max(self.raw_data-self.org_data)

    def get_nzdval(self,data):
        n = data.shape[0]
        nzd_data = (data - self.mu.repeat(n,1)) / (self.std+self.eps).repeat(n,1)
        return nzd_data

    def get_orgval(self,data):
        n = data.shape[0]
        org_data = data*(self.std+self.eps).repeat(n,1) + self.mu.repeat(n,1)
        return org_data

def get_mdn_training_data(x_min=0, x_max=100, n_train_half=1000, y_max=100, noise_std=1.0):
    """
    Get training data for mixture density networks
    """
    # Get x
    x_train = torch.linspace(x_min, x_max, n_train_half).reshape(-1,1)

    # Shuffle
    n = x_train.shape[0]
    random_idx = sample(range(n), n)
    x_train = x_train[random_idx,:]

    # Get y
    sin_ = y_max*torch.sin(2.0*torch.Tensor([math.pi])*x_train/(x_max-x_min))
    cos_ = y_max*torch.cos(2.0*torch.Tensor([math.pi])*x_train/(x_max-x_min))
    y_train = torch.cat([sin_+2*y_max*x_train/x_max, cos_+2*y_max*x_train/x_max], dim=1) # [1000,2]

    x_train,y_train = torch.cat([x_train,x_train],dim=0),torch.cat([y_train,-y_train],dim=0)
    n_train = y_train.shape[0]
    
    # Add noise
    noise = noise_std * y_max * torch.rand(n_train, 2) * torch.square(1 - x_train / x_max)
    y_train = y_train + noise
    
    # Normalize
    nzr_x_train = NormalizerClass(x_train)
    x_train = nzr_x_train.get_nzdval(x_train)
    y_train = NormalizerClass(y_train).nzd_data

    return x_train,y_train

"""
data plotting tool
"""
def plot_mdn_data(noise_std=1.0):
    x_train, y_train = get_mdn_training_data(noise_std=noise_std)

    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(x_train,y_train[:,0],'ro')
    plt.title('+sin & -sin')
    plt.xlabel('Input')
    plt.ylabel('Target')

    plt.subplot(1,2,2)
    plt.plot(x_train, y_train[:, 1], 'ro')
    plt.title('+cos & -cos')
    plt.xlabel('Input')
    plt.ylabel('Target')

    # plt.subplot(2,2,1)
    # plt.plot(x_train[:1000],y_train[:1000,0],'ro')
    # plt.title('+sin')
    # plt.xlabel('Input')
    # plt.ylabel('Target')
    # plt.subplot(2,2,2)
    # plt.plot(x_train[:1000],y_train[:1000,1],'ro')
    # plt.title('+cos')
    # plt.xlabel('Input')
    # plt.ylabel('Target')
    # plt.subplot(2,2,3)
    # plt.plot(x_train[1000:],y_train[1000:,0],'ro')
    # plt.title('-sin')
    # plt.xlabel('Input')
    # plt.ylabel('Target')
    # plt.subplot(2,2,4)
    # plt.plot(x_train[1000:],y_train[1000:,1],'ro')
    # plt.title('-cos')
    # plt.xlabel('Input')
    # plt.ylabel('Target')

    plt.show()

    # save img file in current folder
    plt.savefig("mdn_plot_data.png")

def save_model(model, epoch, PATH, name):
    date = datetime.datetime.now()
    path = os.path.join(PATH, "{}_{}_{}-{}-{}-{}-{}.pth".format(name, epoch, date.year, date.month, date.day, date.hour, date.minute))
    torch.save(model.state_dict(), path)

"""
parameter checking tool
"""
def get_params_dict(model):
    keys = list(dict(model.state_dict()))
    values = list(model.parameters())
    return dict(zip(keys, values))

def print_params(model, keys=None, grad=False, print_keys=True): # keys: list of param_key(str)
    if keys==None:
        keys = get_params_dict(model).keys()
    for key in keys:
        if grad:
            name = key + ".grad"
            val = get_params_dict(model)[key].grad
        else:
            name = key
            val = get_params_dict(model)[key]
        if print_keys:
            print(name)
        print(val)

def print_mog_params(mdn, x_data, data_idx):
    for idx, pi_mean_var in enumerate(mdn(x_data)[:,:,data_idx,:]): # pi_mean_var: [3,k]
        pi, mean, var = pi_mean_var # pi, mean, var: [k,]
        print("-" * 60)
        print("Predicted MoG parameters of x_data[{}] for y_dim[{}]".format(data_idx, idx))
        print("-" * 60)
        print("(pi)")
        print(pi)
        print("(mean)")
        print(mean)
        print("(var)")
        print(var)
        print()

"""
prediction plotting tool
"""
def plot_pred(x_data, y_data, sample, mog_params, pi_th=0.1):
    # x_data: [N x 1], y_data: [N x D], sample: [N x D], MoG_params: [D x 3 x N x k]
    N = x_data.shape[0] # N
    y_dim = mog_params.shape[0] # D
    k = mog_params.shape[3] # k

    # shuffled data -> sorted data
    sort_idx = np.argsort(x_data[:,0])
    x_data = x_data[sort_idx]
    y_data = y_data[sort_idx]
    sample = sample[sort_idx]

    x_max = np.max(x_data)
    x_min = np.min(x_data)

    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(ii) for ii in np.linspace(0, 1, k)]

    # plot
    plt.figure(figsize=(12*y_dim,8))
    for dim, pi_mean_var in enumerate(mog_params): # pi_mean_var: [3 x N x k]
        plt.subplot(1,y_dim,dim+1)
        plt.xlim(x_min, x_max)
        plt.ylim(-3.0,3.0)
        plt.title("[{}]-th dimension".format(dim+1))
        plt.xlabel("Input")
        plt.ylabel("Output")

        # plot y_data
        plt.plot(x_data[:,0], y_data[:,dim],'k.')
        # plot sample
        plt.plot(x_data[:, 0], sample[:,dim], 'rx')

        pi, mean, var = pi_mean_var # each [N x k]
        pi = pi[sort_idx]
        mean = mean[sort_idx]
        var = var[sort_idx]
        for mix_idx in range(k):
            pi_high_idx = np.where(pi[:, mix_idx] > pi_th)[0]  # [?,]
            pi_low_idx = np.where(pi[:, mix_idx] <= pi_th)[0]  # [?,]
            # plot mean
            plt.plot(x_data[pi_high_idx,0], mean[pi_high_idx,mix_idx],
                     '-', color=colors[mix_idx], linewidth=3, label="mix{}".format(mix_idx+1))
            plt.plot(x_data[pi_low_idx,0], mean[pi_low_idx,mix_idx],
                     '-', color=[0.8,0.8,0.8], linewidth=1)
            # plot variance
            upper_bound = mean[:,mix_idx] + 2 * np.sqrt(var[:,mix_idx]) # [N,]
            lower_bound = mean[:,mix_idx] - 2 * np.sqrt(var[:,mix_idx]) # [N,]
            plt.fill_between(x_data[pi_high_idx, 0], lower_bound[pi_high_idx], upper_bound[pi_high_idx],
                             facecolor=colors[mix_idx], interpolate=True, alpha=0.3)
            plt.fill_between(x_data[pi_low_idx, 0], lower_bound[pi_low_idx], upper_bound[pi_low_idx],
                             facecolor=colors[mix_idx], interpolate=True, alpha=0.01)
        plt.legend(fontsize=10)
    # save
    plt.savefig("mdn_plot_prediction.png")

"""
uncertainty plotting tool
"""
def scale_vec(v):
    v_min, v_max = np.min(v), np.max(v)
    return (v - v_min) / (v_max - v_min)

def plot_unct(x_data, y_dim, exp_var, unexp_var):
    x_max = np.max(x_data)
    x_min = np.min(x_data)

    plt.figure(figsize=(12 * y_dim, 8))
    for dim in range(y_dim):
        plt.subplot(1,y_dim,dim+1)
        plt.title("[{}]-th dimension".format(dim+1))
        plt.xlabel("Input")
        plt.ylabel("Uncertainty")
        # shuffled data -> sorted data
        idx_sort = np.argsort(x_data)
        x_data = x_data[idx_sort]
        exp_var = exp_var[:,idx_sort]
        unexp_var = unexp_var[:,idx_sort]
        # plot
        plt.plot(x_data, scale_vec(exp_var[dim]), 'b-', label="Explained") # [N,], [N,]
        plt.plot(x_data, scale_vec(unexp_var[dim]), 'r-', label="Unexplained") # [N,], [N,]
        plt.legend(fontsize=15)
        plt.xlim(x_min, x_max)
    # save
    plt.savefig("mdn_plot_uncertainty.png")