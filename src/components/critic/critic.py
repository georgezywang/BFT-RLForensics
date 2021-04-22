import torch as th
import torch.nn as nn

from module.agents.rnn_agent import RNNAgent


class Critic(nn.Module):
    def __init__(self, scheme, args):
        super(Critic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.rnn = RNNAgent(input_shape, 2, args)

    def init_hidden(self):
        return self.rnn.init_hidden()

    def forward(self, batch, hidden_states, t):
        inputs = self._build_inputs(batch, t)
        return self.rnn(inputs, hidden_states)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []

        # observation
        inputs.append(batch["attacker_obs"][:, t])
        inputs.append(batch["identifier_obs"][:, t])

        # actions
        inputs.append(batch["attacker_action"][:, t])
        inputs.append(batch["identifier_action"][:, t])

        # turn list inputs into tensor array
        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        return scheme["attacker_obs"]["vshape"] + scheme["identifier_obs"]["vshape"] + scheme["attacker_action"]["vshape"] + scheme["identifier_action"]["vshape"]