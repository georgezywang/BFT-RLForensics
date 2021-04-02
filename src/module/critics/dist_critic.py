import torch
import torch.nn.functional as F
from torch import nn

from module.utils.components import MLP
from utils.utils import identity, fanin_init


class DistCritic(nn.Module):
    def __init__(self, scheme, args):
        super(DistCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape()
        output_shape = 1
        self.output_type = "q"

        self.critic = MLP(
            hidden_sizes=args.critic_hidden_sizes,
            input_size=input_shape,
            output_size=output_shape,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_params=None,
        )

    def forward(self, batch, device):
        inputs = self._build_inputs(batch, device)
        bs = batch["z_p"].shape[0]
        # print("sc control critic actual input:{}".format(inputs.shape))
        return self.critic(inputs).reshape(bs)

    def _build_inputs(self, batch, device):
        # assume latent_state: [bs, latent_state_size]
        # obs: [bs, seq_len, n_agents, obs_size]
        bs = batch["z_q"].shape[0]
        inputs = []

        z_p_s = batch["z_p"]  # [bs, n_agents, space_dim]
        z_q_s = batch["z_q"]  # [bs, n_agents, space_dim]
        inputs.append(z_q_s)
        inputs.append(z_p_s)

        # agent_id
        # agent_id = torch.eye(self.n_agents, device=device).unsqueeze(0).expand(bs, -1, -1)
        # inputs.append(agent_id)

        inputs = torch.cat([x.reshape(bs, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self):
        return self.args.latent_relation_space_dim * 2  # z_q, z_q