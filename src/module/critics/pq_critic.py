import torch
import torch.nn.functional as F
from torch import nn

from module.utils.components import MLP
from utils.utils import identity, fanin_init


class PQCritic(nn.Module):
    def __init__(self, scheme, args):
        super(PQCritic, self).__init__()

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

    def forward(self, batch, t):
        inputs = self._build_inputs(batch, t)
        bs = batch.batch_size
        # print("sc control critic actual input:{}".format(inputs.shape))
        return self.critic(inputs).reshape(bs, self.n_agents)

    def _build_inputs(self, batch, t):
        # assume latent_state: [bs, latent_state_size]
        # obs: [bs, seq_len, n_agents, obs_size]
        bs = batch.batch_size
        ts = slice(t, t + 1)
        inputs = []

        # keys, queries and rules
        p_s = batch["p"]  # [bs, n_agents, n_agents]
        q_s = batch["q"]  # [bs, n_agents, n_agents]
        inputs.append(p_s)
        inputs.append(q_s)

        # agent_id
        agent_id = torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
        inputs.append(agent_id)

        inputs = torch.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self):
        return self.n_agents * 3  # p, q, agent_id