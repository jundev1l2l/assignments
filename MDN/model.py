import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class LinearAct(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(LinearAct, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_feat, out_feat)),
            ('act', nn.Tanh()),
        ]))

    def forward(self, x):
        return self.fc(x)

"""
N: batch_size
D: y_dimension(y_dim)
k: num_mixtures in MoG
A: var_max(sig_max)
"""
class MoGOut(nn.Module):
    def __init__(self, in_feat, y_dim, k, A):
        super(MoGOut, self).__init__()
        self.mog = nn.ModuleList()
        """
        modules: y_dim number of (fc_pi, fc_mean, fc_var) layers
        """
        for _ in range(y_dim):
            fc_pi, fc_mean, fc_var = self.get_layers(in_feat, k)
            module = nn.ModuleDict(OrderedDict({"fc_pi":fc_pi, "fc_mean":fc_mean, "fc_var":fc_var})) # 3 layers
            self.mog.append(module) # y_dim modules

        self.A = A
        self.y_dim = y_dim
        self.k = k

    def get_layers(self, in_feat, k):
        fc_pi = nn.Linear(in_feat, k)  # [N,k]
        fc_mean = nn.Linear(in_feat, k)  # [N,k]
        fc_var = nn.Linear(in_feat, k)  # [N,k]
        return fc_pi, fc_mean, fc_var

    def forward(self, x):
        out = torch.empty(self.y_dim, 3, x.size(0), self.k) # [D,3,N,k]
        for dim, module in enumerate(self.mog): # module: nn.ModuleDict for each y_dim
            fc_pi, fc_mean, fc_var = module.values()
            pi = F.softmax(fc_pi(x), dim=-1)  # [N,k]
            mean = fc_mean(x)  # [N,k]
            """
            heuristic: use sigmoid with max_value A
            """
            if self.A == 0:
                var = torch.exp(fc_var(x))  # [N,k]
            else:
                var = self.A * torch.sigmoid(fc_var(x))  # [N,k]
            pi_mean_var = torch.stack([pi, mean, var], dim = 0) # [3,N,k]
            out[dim] = pi_mean_var

        return out # [D,3,N,k]


class MDN(nn.Module):

    def __init__(self, name='MDN', x_dim=1, y_dim=2, k=10, hidden_dims=[32, 32], actv=nn.Tanh, var_max=0,
                 mean_min=-3.0, mean_max=3.0, DEPENDENT_OUTPUT=True):
        """
        Initialize
        """
        super(MDN, self).__init__()

        self.name = name
        self.x_dim = x_dim
        self.y_dim = y_dim # D
        self.k = k  # num_mixtures in mog
        self.hidden_dims = hidden_dims
        self.actv = actv
        self.var_max = var_max

        self.mean_min = mean_min
        self.mean_max = mean_max

        self.DEPENDENT_OUTPUT = DEPENDENT_OUTPUT

        self.layers = nn.ModuleList()
        self.build_model()
        self.weight_init()

    def build_model(self):
        """
        model structure = (LinearAct) * num_hid + GMMOut
        """
        in_feat = self.x_dim
        # fc_layers producing features for out_layer
        for _, out_feat in enumerate(self.hidden_dims):
            hidden_layer = LinearAct(in_feat, out_feat)
            self.layers.append(hidden_layer)
            in_feat = out_feat
        # final output layer producing GMM parameters pi, mean, var
        out_layer = MoGOut(in_feat, self.y_dim, self.k, self.var_max)
        self.layers.append(out_layer)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        """   
        heuristic: fc_mean.bias ~ Uniform(mean_min, mean_max)
        """
        for idx in range(self.y_dim):
            self.layers[-1].mog[idx].fc_mean.bias.data.uniform_(self.mean_min, self.mean_max)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # [D,3,N,k]