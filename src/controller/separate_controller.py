import torch as th

from components.action_selectors import EpsilonGreedyAttackerActionSelector, EpsilonGreedyIdentifierActionSelector
from module.agents.rnn_agent import RNNIdentifierAgent, RNNAttackerAgent


class SeparateMAC:
    def __init__(self, scheme, groups, args):
        self.n_peers = args.n_peers
        self.args = args
        self.scheme = scheme
        self.attacker = None
        self.identifier = None
        self.attacker_hidden_states = None
        self.identifier_hidden_states = None
        self._build_agents()
        self.attacker_action_selector = EpsilonGreedyAttackerActionSelector(args)
        self.identifier_action_selector = EpsilonGreedyIdentifierActionSelector(args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        attacker_outputs, identifier_outputs = self.forward(ep_batch, t_ep, test_mode)
        attacker_chosen_actions = self.attacker_action_selector.select_action(attacker_outputs[bs], t_env,
                                                                              test_mode=test_mode)
        identifier_chosen_actions = self.identifier_action_selector.select_action(identifier_outputs[bs], t_env,
                                                                                  test_mode=test_mode)
        return attacker_chosen_actions, identifier_chosen_actions

    def forward(self, ep_batch, t, test_mode):
        attacker_input, identifier_input = self._build_inputs(ep_batch, t)
        attacker_outs, self.attacker_hidden_states = self.attacker(attacker_input, self.attacker_hidden_states)
        identifier_outs, self.identifier_hidden_states = self.identifier(identifier_input, self.identifier_hidden_states)

        # process the attacker
        if not test_mode:
            # Epsilon floor for attacker
            exploring_attacker_outs = []
            for out in attacker_outs[:-1]:
                epsilon_action_num = out.size(-1)
                out = out.view(-1, epsilon_action_num)
                exploring_out = ((1 - self.attacker_action_selector.epsilon) * out
                                 + th.ones_like(out) * self.attacker_action_selector.epsilon/epsilon_action_num)
                exploring_out = exploring_out.view(ep_batch.batch_size, self.args.max_message_num_per_round, -1)
                exploring_attacker_outs.append(exploring_out)

            cert_outs = []
            for idx in range(self.n_peers):  # attacker_outs[-1]: ([bs, max_msg_num, 2])*n_peers
                cert_out = attacker_outs[-1][idx]
                epsilon_action_num = 2
                cert_out = cert_out.view(-1, epsilon_action_num)
                exploring_cert_out = ((1 - self.attacker_action_selector.epsilon) * cert_out
                                 + th.ones_like(cert_out) * self.attacker_action_selector.epsilon / epsilon_action_num)
                exploring_cert_out = exploring_cert_out.view(ep_batch.batch_size, self.args.max_message_num_per_round, -1)
                cert_outs.append(exploring_cert_out)

            exploring_attacker_outs.append(cert_outs)

            # epsilon floor for identifier
            epsilon_action_num = 2
            exploring_identifier_outs = ((1 - self.identifier_action_selector.epsilon) * identifier_outs
                                         + th.ones_like(identifier_outs) * self.identifier_action_selector.epsilon / epsilon_action_num)
            return exploring_attacker_outs, exploring_identifier_outs
        else:
            return attacker_outs, identifier_outs

    def init_hidden(self, batch_size):
        if 'init_hidden' in self.attacker.__dict__:
            self.attacker_hidden_states = self.attacker.init_hidden().unsqueeze(0).expand(batch_size, -1)  # bav
        if 'init_hidden' in self.identifier.__dict__:
            self.identifier_hidden_states = self.identifier.init_hidden().unsqueeze(0).expand(batch_size, -1)

    def attacker_parameters(self):
        return list(self.attacker.parameters())

    def identifier_parameters(self):
        return list(self.identifier.parameters())

    def parameters(self):
        return list(self.attacker.parameters()) + list(self.identifier.parameters())

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
        self.attacker = RNNAttackerAgent(input_shape_attacker, output_shape_attacker, self.args)
        self.identifier = RNNIdentifierAgent(input_shape_identifier, output_shape_identifier, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent

        attacker_inputs = [batch["attacker_obs"][:, t]]
        attacker_inputs = th.cat(attacker_inputs, dim=1)

        identifier_inputs = [batch["identifier_obs"][:, t]]
        identifier_inputs = th.cat(identifier_inputs, dim=1)

        return attacker_inputs, identifier_inputs

    def _get_input_shape(self, scheme):
        attacker_input_shape = scheme["attacker_obs"]["vshape"]
        identifier_input_shape = scheme["identifier_obs"]["vshape"]
        return attacker_input_shape, identifier_input_shape

    def _get_output_shape(self, scheme):
        num_msg_type = 10  # no client type, 9 is no-op
        msg_action_space = num_msg_type + self.args.num_malicious + \
                           self.args.max_seq_num + self.args.max_view_num + \
                           self.args.total_client_vals + self.args.n_peers + self.args.n_peers * 2
        attacker_output_shape = msg_action_space * self.args.max_message_num_per_round * self.args.num_malicious
        identifier_output_shape = self.n_peers
        return attacker_output_shape, identifier_output_shape
