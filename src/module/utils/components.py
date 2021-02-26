import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.utils import identity, fanin_init, product_of_gaussians, LayerNorm


class MLPMultiGaussianEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 mlp_hidden_sizes,
                 mlp_init_w=3e-3,
                 mlp_hidden_activation=F.relu,
                 mlp_output_activation=identity,
                 mlp_hidden_init=fanin_init,
                 mlp_bias_init_value=0.1,
                 mlp_layer_norm=False,
                 mlp_layer_norm_params=None,
                 use_information_bottleneck=True,
                 ):
        super(MLPMultiGaussianEncoder, self).__init__()
        self.mlp = FlattenMLP(
            input_size=input_size,
            output_size=2*output_size if use_information_bottleneck else output_size,  # vars + means
            hidden_sizes=mlp_hidden_sizes,
            init_w=mlp_init_w,
            hidden_activation=mlp_hidden_activation,
            output_activation=mlp_output_activation,
            hidden_init=mlp_hidden_init,
            b_init_value=mlp_bias_init_value,
            layer_norm=mlp_layer_norm,
            layer_norm_params=mlp_layer_norm_params,
        )
        self.use_information_bottleneck = use_information_bottleneck
        self.input_size = input_size
        self.output_size = output_size
        self.z_means = None
        self.z_vars = None

    def infer_posterior(self, inputs):
        self.forward(inputs)
        return self.z

    def sample_z(self):
        if self.use_information_bottleneck:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in
                          zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def forward(self, input):
        params = self.mlp(input)  #[batch_size, 2*output_size]
        if self.use_information_bottleneck:
            self.z_means = params[..., :self.output_size]
            self.z_vars = F.softplus(params[..., self.output_size:])
            # z_params = [product_of_gaussians(m, s) for m,s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        else:
            self.z_means = torch.mean(params, dim=1)  # FIXME: doublecheck
            self.z_vars = None
        self.sample_z()

    def compute_kl_div(self):
        prior = torch.distributions.Normal(torch.zeros(self.output_size), torch.ones(self.output_size))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in
                      zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def reset(self):
        self.z_means = None
        self.z_vars = None


class MLP(nn.Module):
    # https://github.com/katerakelly/oyster/blob/master/rlkit/torch/networks.py
    def __init__(self,
                 hidden_sizes,
                 input_size,
                 output_size,
                 init_w=3e-3,
                 hidden_activation=F.relu,
                 output_activation=identity,
                 hidden_init=fanin_init,
                 b_init_value=0.1,
                 layer_norm=False,
                 layer_norm_params=None,
                 ):
        super(MLP, self).__init__()
        if layer_norm_params is None:
            layer_norm_params = dict()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm

        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        # self.last_fc.weight.data.uniform_(-init_w, init_w)
        # self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivation=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivation:
            return output, preactivation
        else:
            return output

class FlattenMLP(MLP):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)

class MLPEncoder(FlattenMLP):
    def reset(self, num_task=1):
        pass