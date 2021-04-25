"""
Code adapted from https://github.com/TonghanWang/ROMA
"""
import torch
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            print((torch.gather(avail_actions, dim=2,
                                index=masked_q_values.max(dim=2)[1].unsqueeze(2)) <= 0.99).squeeze())
            print((torch.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) <= 0.99).squeeze())

            print('Action Selection Error')
            # raise Exception
            return self.select_action(agent_inputs, avail_actions, t_env, test_mode)

        return picked_actions


class EpsilonGreedyAttackerActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        bs = agent_inputs[0].size(0)
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        picked = []  # HUH, probably everythin here is not necessary
        for input_q_vals in agent_inputs[:-1]:
            num_choices = input_q_vals.size(-1)
            random_numbers = torch.rand_like(input_q_vals[:, :, 0]).to("cpu")
            pick_random = (random_numbers < self.epsilon).long()
            probs = torch.tensor([1 / num_choices] * num_choices)
            choices = input_q_vals.max(dim=2)[1]
            random_actions = Categorical(probs).sample(choices.shape).long()
            # print(random_actions.is_cuda)
            # print(choices.is_cuda)
            picked_actions = pick_random * random_actions + (1 - pick_random) * choices.to("cpu")
            picked.append(torch.eye(num_choices)[picked_actions])
            # print(num_choices)

        picked_sigs = []
        for c_id in range(self.args.n_peers):  # ([bs, max_msg_num, 2])*n_peers
            num_choices = agent_inputs[-1][c_id].size(-1)  # 2
            random_numbers = torch.rand_like(agent_inputs[-1][c_id][:, :, 0]).to("cpu")
            pick_random = (random_numbers < self.epsilon).long()
            probs = torch.tensor([1 / num_choices] * num_choices)
            choices = agent_inputs[-1][c_id].max(dim=2)[1]
            random_actions = Categorical(probs).sample(choices.shape).long()
            picked_actions = pick_random * random_actions + (1 - pick_random) * choices.to("cpu")
            picked_sigs.append(torch.eye(num_choices)[picked_actions])

        picked_sigs = torch.cat(picked_sigs, dim=-1)

        picked.append(picked_sigs)
        # print("picked action: {}".format([x.shape for x in picked]))
        picked = torch.cat(picked, dim=-1)
        # print("action shape: {}".format(picked.shape))
        return picked.view(bs, -1)  # [bs, max_num_msg_per_round, msg_space]


class EpsilonGreedyIdentifierActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        num_choices = 2
        random_numbers = torch.rand_like(agent_inputs)
        pick_random = (random_numbers < self.epsilon).long()
        probs = torch.tensor([1 / num_choices] * num_choices)
        random_actions = Categorical(probs).sample(agent_inputs.shape).long().to(device)
        picked_actions = pick_random * random_actions + (1 - pick_random) * agent_inputs.ge(0.5).long()

        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

