"""
Code adapted from https://github.com/TonghanWang/ROMA
"""
import torch
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}

class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = torch.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        if not (torch.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99).all():
            print((torch.gather(avail_actions, dim=2, index=random_actions.unsqueeze(2)) <= 0.99).squeeze())
            print((torch.gather(avail_actions, dim=2, index=masked_q_values.max(dim=2)[1].unsqueeze(2)) <= 0.99).squeeze())
            print((torch.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) <= 0.99).squeeze())

            print('Action Selection Error')
            # raise Exception
            return self.select_action(agent_inputs, avail_actions, t_env, test_mode)

        return picked_actions

    def select_pq_values(self, raw_pq, t_env, test_mode=False):
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        random_numbers = torch.rand_like(raw_pq, dtype=torch.float)
        pick_random = (random_numbers < self.epsilon).long()
        prob = torch.tensor([0.5, 0.5])
        random_pq = torch.tensor((Categorical(prob).sample().long(), Categorical(prob).sample().long()))
        picked_pq = pick_random * random_pq + (1 - pick_random) * raw_pq
        return picked_pq



REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
