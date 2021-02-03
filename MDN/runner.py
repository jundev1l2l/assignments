import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributions as D

from random import sample
from utils import save_model, plot_pred, plot_unct


def get_mog(pi, mean, var): # each [N,k]
    mix_weights = D.Categorical(probs=pi)
    mix_dists = D.Normal(loc=mean,scale=torch.sqrt(var)) # scale: standard deviation
    mix = D.MixtureSameFamily(mix_weights, mix_dists)
    return mix # sample -> [N,]

def get_sample(mdn, x_data):
    samples = torch.empty(x_data.size(0), mdn.y_dim) # [N,D]
    for idx, pi_mean_var in enumerate(mdn(x_data)): # pi_mean_var: [3,N,k]
        pi, mean, var = pi_mean_var # pi, mean, var: [N,k]
        mog = get_mog(pi, mean, var)
        samples[:,idx] = mog.sample() # [N,]
    return samples # [N,D]

def get_exp_var(pi, mean, var):
    k = pi.size(1)
    mean_average = torch.sum(torch.mul(pi, mean), dim=-1).unsqueeze(dim=-1)  # [N,1]
    diff_squared = torch.square(mean - mean_average.repeat(1, k))  # [N,k]
    exp_vars = torch.sum(torch.mul(pi, diff_squared), dim=-1)  # [N,]
    return exp_vars # [N,]

def get_unexp_var(pi, mean, var):
    unexp_vars = torch.sum(torch.mul(var, pi), dim=-1)  # [N,]
    return unexp_vars # [N,]

def get_unct_dict(mdn, x_data):
    exp_vars = torch.empty(mdn.y_dim, x_data.size(0)) # [D,N]
    unexp_vars = torch.empty(mdn.y_dim, x_data.size(0)) # [D,N]
    for dim, pi_mean_var in enumerate(mdn(x_data)):  # pi_mean_var: [3,N,k]
        pi, mean, var = pi_mean_var # pi, mean, var: [N,k]
        exp_vars[dim] = get_exp_var(pi, mean, var)
        unexp_vars[dim] = get_unexp_var(pi, mean, var)
    return {"exp": exp_vars, "unexp": unexp_vars} # {"exp": [D,N], "unexp": [D,N]}


class MDNRunner():
    def __init__(self, model, optimizer=optim.RMSprop, batch_size = 128, lr=1e-3, weight_decay=1e-8, exp_var_decay=1e-6):
        self.MDN = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.exp_var_decay = exp_var_decay

    def set_data(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def train(self, epochs=5000, save=False, loss_every=1000):
        print("Start Training")
        print("batch_size:{} learning_rate:{} weight_decay:{} exp_var_decay:{}".format(self.batch_size,self.lr,self.weight_decay,self.exp_var_decay))
        print("Epoch     Y-dim      Loss     Neg_Lik_Loss     Exp_Var_Loss")
        optimizer = self.optimizer(params=self.MDN.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        for epoch in range(epochs):
            # initialize
            losses = []
            loss = 0
            optimizer.zero_grad()
            # random batch sampling
            r_idx = sample(range(self.x_data.size(0)), self.batch_size)
            x_batch, y_batch = self.x_data[r_idx, :], self.y_data[r_idx, :]
            # evaluate
            for idx, pi_mean_var in enumerate(self.MDN(x_batch)): # pi_mean_var: [3,N,k]
                pi, mean, var = pi_mean_var # pi, mean, var: [N,k]
                mog = get_mog(pi, mean, var)

                nll_loss = - torch.mean(mog.log_prob(y_batch[:,idx])) # [1,] negative log likelihood loss

                exp_vars = get_exp_var(pi, mean, var)  # [N,]
                """
                heuristic: use exp_var_loss to maximize exp_var
                """
                exp_var_loss = - self.exp_var_decay * torch.sum(exp_vars, axis=-1)  # [1,] epstemic uncertainty regularization loss

                total_loss = nll_loss + exp_var_loss
                loss += total_loss
                losses += [(total_loss.item(),nll_loss.item(),exp_var_loss.item())] # will become [D,]
            # update
            loss.backward()
            optimizer.step()
            # print
            if epoch % loss_every == (loss_every - 1):
                print("{:4d}".format(epoch + 1), end=" " * 8)
                for idx, loss in enumerate(losses):
                    print("{}      {:.4f}      {:.4f}         {:.4f}e-4".format(idx, loss[0], loss[1], loss[2] * 1e4),
                          end=" " * 10)
                print()
        print("Training Finished")
        # save
        if save:
            save_model(self.MDN, epochs, "/Users/junhyunpark1995/Documents/GitHub/pBus-MDN/saved_model", "mdn")
            print("Trained Model Saved")

def plot_prediction(mdn, x_data, y_data, pi_th=0.1):
    """
    Plot prediction
    """
    sample = get_sample(mdn, x_data)
    plot_pred(x_data.detach().numpy(), # [N,1]
              y_data.detach().numpy(), # [N,D]
              sample.detach().numpy(), # [N,D]
              mdn(x_data).detach().numpy(), # [D,3,N,k]
              pi_th) # pi_threshold

def plot_uncertainty(mdn, x_data):
    """
    Plot uncertainty with prediction
    """
    uncts = get_unct_dict(mdn, x_data) # dict
    plot_unct(x_data[:,0].numpy(), # x_data: [N,]
              mdn.y_dim, # D
              exp_var=uncts["exp"].detach().numpy(), # [D,N]
              unexp_var=uncts["unexp"].detach().numpy()) # [D,N]