import torch as th

from components.action_selectors import EpsilonGreedyActionSelector
from module.agents import REGISTRY as agent_REGISTRY


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.scheme = scheme
        self._build_agents()
        self.attacker_action_selector = EpsilonGreedyActionSelector(args)
        self.identifier_action_selector = EpsilonGreedyActionSelector(args)

        self.attacker_hidden_states = None
        self.identifier_hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        attacker_avail_actions = ep_batch["attacker_avail_actions"][:, t]
        identifier_avail_actions = ep_batch["identifier_avail_actions"][:, t]
        attacker_outputs, identifier_outputs = self.forward(ep_batch, t_ep)
        attacker_chosen_actions = self.attacker_action_selector.select_action(attacker_outputs[bs], attacker_avail_actions[bs], t_env, test_mode=test_mode)
        identifier_chosen_actions = self.identifier_action_selector.select_action(identifier_outputs[bs], identifier_avail_actions[bs], t_env, test_mode=test_mode)
        return attacker_chosen_actions, identifier_chosen_actions

    def forward(self, ep_batch, t):
        attacker_input, identifier_input = self._build_inputs(ep_batch, t)
        attacker_outs, self.attacker_hidden_states = self.attacker(attacker_input, self.attacker_hidden_states)
        identifier_outs, self.identifier_hidden_states = self.identifier(identifier_input, self.identifier_hidden_states)
        return attacker_outs.view(ep_batch.batch_size, -1), identifier_outs.view(ep_batch.batch_size, -1)

    def compute_equilibrium_payoffs(self, ep_batch, t):


    def init_hidden(self, batch_size):
        if 'init_hidden' in self.attacker.__dict__:
            self.attacker_hidden_states = self.attacker.init_hidden().unsqueeze(0).expand(batch_size, -1)  # bav
        if 'init_hidden' in self.identifier.__dict__:
            self.identifier_hidden_states = self.identifier.init_hidden().unsqueeze(0).expand(batch_size, -1)

    def parameters(self):
        return list(self.attacker.parameters())+list(self.identifier.parameters())

    def load_state(self, other_mac):
        self.attacker.load_state_dict(other_mac.attacker.state_dict())
        self.identifier.load_state_dict(other_mac.identifier.state_dict())

    def cuda(self):
        self.attacker.cuda()
        self.identifier.cuda()

    def save_models(self, path):
        th.save(self.attacker.state_dict(), "{}/attacker.th".format(path))
        th.save(self.identifier.state_dict(), "{}/identifier.th".format(path))

    def load_models(self, path):
        self.attacker.load_state_dict(th.load("{}/attacker.th".format(path), map_location=lambda storage, loc: storage))
        self.identifier.load_state_dict(th.load("{}/identifier.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self):
        input_shape_attacker, input_shape_identifier = self._get_input_shape(self.scheme)
        output_shape_attacker, output_shape_identifier = self._get_output_shape(self.scheme)
        self.attacker = agent_REGISTRY[self.args.agent_attacker](input_shape_attacker, output_shape_attacker, self.args)
        self.identifier = agent_REGISTRY[self.args.agent_identifier](input_shape_identifier, output_shape_identifier, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size

        attacker_inputs = []
        attacker_inputs.append(batch["attacker_obs"][:, t])  # b1av
        # observe opponent's last action
        if t == 0:
            attacker_inputs.append(th.zeros_like(batch["identifier_actions_onehot"][:, t]))
        else:
            attacker_inputs.append(batch["identifier_actions_onehot"][:, t-1])
        attacker_inputs = th.cat(attacker_inputs, dim=1)

        identifier_inputs = []
        identifier_inputs.append(batch["identifier_obs"][:, t])  # b1av
        # observe opponent's last action
        if t == 0:
            identifier_inputs.append(th.zeros_like(batch["attacker_actions_onehot"][:, t]))
        else:
            identifier_inputs.append(batch["attacker_actions_onehot"][:, t - 1])
        identifier_inputs = th.cat(identifier_inputs, dim=1)

        return attacker_inputs, identifier_inputs

    def _build_payoff_table_inputs(self, ep_batch, t):
        # [agents][]
        bs = ep_batch.batch_size
        inputs = []


    def _get_input_shape(self, scheme):
        attacker_input_shape = scheme["attacker_obs"]["vshape"] + scheme["identifier_actions_onehot"]["vshape"][0]
        identifier_input_shape = scheme["identifier_obs"]["vshape"] + scheme["attacker_actions_onehot"]["vshape"][0]
        return attacker_input_shape, identifier_input_shape

    def _get_output_shape(self, scheme):
        attacker_output_shape = scheme["attacker_actions_onehot"]
        identifier_output_shape = scheme["identifier_actions_onehot"]