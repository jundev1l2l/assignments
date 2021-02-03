import torch.nn as nn

from utils import get_mdn_training_data, print_params, print_mog_params
from model import MDN
from runner import MDNRunner, plot_prediction, plot_uncertainty, get_unct_dict


def main():
    # need argparsing (from utils.py)

    """
    data config
    """
    x_min = 0              # tf code: 0
    x_max = 100            # tf code: 100
    n_train_half = 1000    # tf code: 1000
    y_max = 100            # tf code: 100
    noise_std = 1.0        # tf code: 1.0

    """
    model config
    """
    name = 'MDN'
    x_dim = 1
    y_dim = 2
    k = 10
    hidden_dims = [32, 32]
    actv = nn.Tanh
    var_max = 0
    mean_min = -3.0
    mean_max = 3.0
    DEPENDENT_OUTPUT = True

    """
    runner config
    """
    epochs = 5000
    loss_every = 1000
    save = False
    batch_size = 128       # tf code: 128
    learning_rate = 1e-3   # tf code: 1e-3
    exp_var_decay = 1e-6   # tf code: 1e-6
    weight_decay = 1e-8    # tf code: 1e-8
    pi_th = 0.1            # tf code: 0.1

    """
    train_data & model
    """
    x_train, y_train = get_mdn_training_data(x_min=x_min,x_max=x_max,n_train_half=n_train_half,y_max=y_max,noise_std=noise_std)
    if y_dim == 1:
        y_train = y_train[:, 0].unsqueeze(dim=-1)
        mdn = MDN(y_dim=1)
    else:
        mdn = MDN()
    print(mdn)

    """
    runner
    
        check out following cases to see how loss function works

        (1) exp_var_decay=1e-6(default) -> nll_loss decreases, epi_loss increses
        (2) exp_var_decay=1e-2 -> nll_loss increases, epi_loss decreases

        but in both cases, (total) loss decreases !!     
    """
    runner = MDNRunner(mdn, batch_size=batch_size, lr=learning_rate, weight_decay = 1e-8, exp_var_decay=exp_var_decay)
    runner.set_data(x_train, y_train)
    runner.train(epochs=epochs, save=save, loss_every=loss_every)

    """
    plotting result
    """
    plot_prediction(mdn, x_train, y_train, pi_th=pi_th)
    plot_uncertainty(mdn, x_train)

    """
    to see initial prediction
    """
    # print_mog_params(mdn, x_data, 0)

    """
    to see learned parameters
    """
    # print_params(mdn)
    # print()

    """
    to see predicted mog_params
    """
    # print_mog_params(mdn, x_data, 0)

    """
    to see uncertainty values
    """
    # unct_dict = get_unct_dict(mdn, x_data)
    # print(unct_dict["exp"])
    # print(unct_dict["unexp"])


if __name__=="__main__":
    main()
