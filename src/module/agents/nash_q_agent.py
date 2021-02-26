from operator import itemgetter

import torch
import torch.nn as nn

from module.agents.rnn_agent import RNNAgent
from module.utils.components import MLP


class NashQAgent(nn.Module):
    """
    Modified Nash Q Agent: (hardcoded solution for 2 agents)

    """

    def __init__(self, args, scheme):
        super(NashQAgent).__init__()
        self.args = args
        self.n_agents = args.n_agents
        step_input_shape, control_input_shape = self._get_input_shapes(scheme)
        step_output_shape, control_output_shape = self._get_output_shapes(scheme)
        self.step_estimator = RNNAgent(step_input_shape, step_output_shape, args)
        self.control_state_estimator = MLP(self.args.control_estimator_hidden_sizes, control_input_shape,
                                           control_output_shape, args)

    def init_hidden(self):
        return self.step_estimator.init_hidden()

    def forward(self, batch, t, hidden_states):
        step_inputs, control_inputs = self._build_inputs(batch, t)
        step_out, step_hidden_states = self.step_estimator(step_inputs)
        control_state_out = self.control_estimator(control_inputs)
        pq, pq_vals = self._choose_pq(control_state_out)
        return step_out, pq, pq_vals, control_state_out, step_hidden_states

    def _choose_pq(self, control_state_out):
        bs = control_state_out.shape[0]
        agent_1_vals = control_state_out[:, 0:4].clone()
        agent_2_vals = control_state_out[:, 4:8].clone()
        agent_1_pq_vals, agent_1_pq_idx = agent_1_vals.max(dim=1)  # [bs]
        agent_2_pq_vals, agent_2_pq_idx = agent_2_vals.max(dim=1)  # [bs]
        pq_dir = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
        agent_1_pq = torch.tensor([pq_dir[int(agent_1_pq_idx[idx])] for idx in range(bs)])  # [bs ,2]
        agent_2_pq = torch.tensor([pq_dir[int(agent_2_pq_idx[idx])] for idx in range(bs)])  # [bs, 2]
        pq = torch.stack([agent_1_pq, agent_2_pq], dim=1)  # [bs, 2, 2]
        pq_vals = torch.stack([agent_1_pq_vals, agent_2_pq_vals], dim=1)  # [bs, 2, 1]
        return pq, pq_vals

    def index_to_pq_vals(self, pq, control_state_out):
        # pq: [bs, n_agents, qp]
        # control: [bs, 8]
        rev_pq_dir = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        bs = pq.shape(0)
        pq_vals = []
        for b_idx in range(bs):
            agent_1_pq = (int(pq[b_idx, 0, 0]), int(pq[b_idx, 0, 1]))
            agent_2_pq = (int(pq[b_idx, 1, 0]), int(pq[b_idx, 1, 1]))
            agent_1_idx = rev_pq_dir[agent_1_pq]
            agent_2_idx = rev_pq_dir[agent_2_pq] + 4
            pq_vals.append(torch.tensor([control_state_out[b_idx][agent_1_idx], control_state_out[b_idx][agent_2_idx]]))
        pq_vals = torch.stack(pq_vals, dim=0)
        return pq_vals  # [bs, 2]

    def compute_equilibrium(self, raw_control_values, raw_step_values):
        # control_state_values: [bs, control_dim*n_agent+n_agent]
        # raw_action_values: [bs, n_actions]
        # returns [bs, num_agent, control_dim]: 0 - p, 1 - q
        bs = raw_control_values.shape[0]
        if self.n_agents != 2 or self.args.control_dim != 2:
            print("Lazy solution not for you ; )")
            return None

        agents_dir = {0: (1, 0), 1: (0, 1)}
        pq_dir = {0: (0, 1), 1: (1, 0)}
        pq_choices = [(1, 1), (1, 0), (0, 1), (0, 0)]  # (p, q)

        step_reward_1_table = [[2, 3, 0, 1],
                               [0, 1, 0, 1],
                               [3, 3, 1, 1],
                               [1, 1, 1, 1]]
        step_reward_2_table = [[1, 0, 1, 2],
                               [3, 2, 3, 2],
                               [0, 0, 2, 2],
                               [2, 2, 2, 2]]
        control_table_index = {[0, 1, 0, 1, 0, 1]: 0,
                               [0, 1, 1, 0, 0, 1]: 1,
                               [1, 0, 0, 1, 0, 1]: 2,
                               [1, 0, 1, 0, 0, 1]: 3,
                               [0, 1, 0, 1, 1, 0]: 4,
                               [0, 1, 1, 0, 1, 0]: 5,
                               [1, 0, 0, 1, 1, 0]: 6,
                               [1, 0, 1, 0, 1, 0]: 7}

        def get_pq_and_control_rewards(batch_idx, pq_choice, agent_id):
            p = pq_choice[0]
            q = pq_choice[1]
            raw_comb = pq_dir[p] + pq_dir[q] + agents_dir[agent_id]
            agent_table_idx = control_table_index[raw_comb]
            agent_control_reward = raw_control_values[batch_idx][agent_table_idx]
            return p, q, agent_control_reward

        def parse_step_table_values(choice1, choice2, c1, c2):
            if step_reward_1_table[choice2][choice1] == 0:
                r1 = 0
            elif step_reward_1_table[choice2][choice1] == 1:
                r1 = c1
            elif step_reward_1_table[choice2][choice1] == 2:
                r1 = c2
            else:
                r1 = c1 + c2

            if step_reward_2_table[choice2][choice1] == 0:
                r2 = 0
            elif step_reward_2_table[choice2][choice1] == 1:
                r2 = c1
            elif step_reward_2_table[choice2][choice1] == 2:
                r2 = c2
            else:
                r2 = c1 + c2
            return r2, r1  # payoff_table[agent_2][agent_1] = (agent_2_payoff, agent_1_paayoff)

        action_values = raw_step_values.reshape(bs, self.n_agents, -1).max(dim=2)[0]  # [bs, n_agents]
        total_choices = 4
        ne = []
        for b_idx in range(bs):
            # builds a payoff table
            payoff_table = [[(0, 0)] * total_choices] * total_choices
            c1 = action_values[b_idx][0]
            c2 = action_values[b_idx][1]
            # paves the step rewards
            for i in range(total_choices):  # agent_2
                for j in range(total_choices):  # agent_1
                    payoff_table[i][j] = parse_step_table_values(j, i, c1, c2)

            # updates agent1's
            for agent_1_idx in range(total_choices):
                agent_1_pq = pq_choices[agent_1_idx]
                p1, q1, con_r1 = get_pq_and_control_rewards(b_idx, agent_1_pq, agent_id=0)
                for agent_2_idx in range(total_choices):
                    payoff_table[agent_2_idx][agent_1_idx][1] += con_r1

            # updates agent2's
            for agent_2_idx in range(total_choices):
                agent_2_pq = pq_choices[agent_2_idx]
                p2, q2, con_r2 = get_pq_and_control_rewards(b_idx, agent_2_pq, agent_id=1)
                for agent_1_idx in range(total_choices):
                    payoff_table[agent_2_idx][agent_1_idx][0] += con_r2

            eq_1_pq_max_val = -1e9
            eq_2_pq_max_val = -1e9

            found = False
            for agent_2_idx in range(total_choices):
                for agent_1_idx in range(total_choices):
                    agent_2_val, agent_1_val = payoff_table[agent_2_idx][agent_1_idx]
                    max_response_1_val = max(payoff_table[agent_2_idx], key=itemgetter(1))[1]
                    response_2_vals = [payoff_table[idx][agent_1_idx][0] for idx in range(total_choices)]
                    max_response_2_val = max(response_2_vals)
                    if max_response_1_val == agent_1_val and max_response_2_val == agent_2_val:
                        found = True
                        # finding the optimal NE
                        if max_response_1_val > eq_1_pq_max_val and max_response_2_val > eq_2_pq_max_val:
                            eq_1_pq_max_val = max_response_1_val
                            eq_2_pq_max_val = max_response_2_val
            if not found:
                print("NE not found!")
                eq_1_pq_max_val, eq_2_pq_max_val = 0, 0
            ne.append(torch.tensor([eq_1_pq_max_val, eq_2_pq_max_val]))

        ne = torch.stack([x for x in ne], dim=0)  # [bs, n_agents]

        return ne  # the value of ne

    def _get_hardcoded_table_input(self, bs):
        if self.n_agents != 2 or self.args.control_dim != 2:
            print("Lazy solution not for you ; )")
            return None
        inputs = [[0, 1, 0, 1, 0, 1],
                  [0, 1, 1, 0, 0, 1],
                  [1, 0, 0, 1, 0, 1],
                  [1, 0, 1, 0, 0, 1],
                  [0, 1, 0, 1, 1, 0],
                  [0, 1, 1, 0, 1, 0],
                  [1, 0, 0, 1, 1, 0],
                  [1, 0, 1, 0, 1, 0]]
        inputs = torch.tensor(inputs).unsqueeze(0).expand(bs, -1, -1)
        return inputs

    def _build_inputs(self, batch, t):
        # Assumes other agents' actions as public knowledge
        # Assumes homogenous agents with flat observations.
        #
        bs = batch.batch_size
        step_inputs, control_state_inputs = [], []
        step_inputs.append(batch["obs"][:, t])  # b1av #
        print(
            "FIXME: All agents' actions need to be public knowledge, current onehot-encoded actions input size: {}".format(
                batch["actions_onehot"].shape))
        if t == 0:
            step_inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
        else:
            step_inputs.append(batch["actions_onehot"][:, t - 1])
        step_inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        step_inputs = torch.cat([x.reshape(bs * self.n_agents, -1) for x in step_inputs], dim=1)

        control_state_inputs = self._get_hardcoded_table_input(bs)
        return step_inputs, control_state_inputs

    def _get_input_shapes(self, scheme):
        # step reward q estimator: obs + actions_onehot (all agents'!) + n_agents
        # pq control state reward q estimator: control_dim + n_agents
        step_reward_input_shape = scheme["obs"]["vshape"] + scheme["actions_onehot"]["vshape"][0] + self.n_agents
        control_state_reward_input_shape = self.args.control_dim * self.n_agents + self.n_agents
        return step_reward_input_shape, control_state_reward_input_shape

    def _get_output_shapes(self, scheme):
        step_reward_output_shape = self.args.n_actions
        control_state_reward_output_shape = 1
        return step_reward_output_shape, control_state_reward_output_shape
