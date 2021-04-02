import torch as th

# This multi-agent controller shares parameters between agents
from module.agents.nash_q_agent import NashQAgent
from components.action_selectors import EpsilonGreedyActionSelector


class SharedNashMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent = NashQAgent(self.args, scheme)
        self.agent_output_type = args.agent_output_type
        self.action_selector = EpsilonGreedyActionSelector(args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # FIXME: chosen actions should be dependent on chosen p qs
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        step_outputs, pq, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(step_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        chosen_pqs = self.action_selector.select_pq_values(pq, t_env, test_mode=test_mode)
        return chosen_actions, chosen_pqs

    def forward(self, ep_batch, t, test_mode=False):
        step_out, pq, pq_vals, control_state_out, self.hidden_states = self.agent(ep_batch, t, self.hidden_states)
        return step_out.view(ep_batch.batch_size, self.n_agents, -1), \
               pq, \
               pq_vals, \
               control_state_out  # no reshape bc of the bad struture path I have taken...

    def index_to_pq_vals(self, pq, control_state_out):
        return self.agent.index_to_pq_vals(pq=pq, control_state_out=control_state_out)

    def compute_equlibrium(self, raw_control_values, raw_step_values):
        return self.agent.compute_equilibrium(raw_control_values=raw_control_values, raw_step_values=raw_step_values)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
