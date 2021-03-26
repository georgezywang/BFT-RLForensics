"""
Code adapted from https://github.com/TonghanWang/ROMA
"""
import copy
import math

from env import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class PQEpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.n_agents = self.args.n_agents
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](self.args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, p, q, test_mode=False, sample_mode=False):  # run one eps
        self.reset()

        terminated = False
        episode_return = [0] * self.n_agents
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            if self.env.is_masked():
                pre_transition_data = {
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [self.env.get_obs()],
                    "adjacent_agents": [self.env.get_adj()],
                    "p": p,
                    "q": q,
                }
            else:
                pre_transition_data = {
                    "state": [self.env.get_state()],
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [self.env.get_obs()],
                    "p": p,
                    "q": q,
                }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # pq: [1, agent_num, control_dim]

            rewards, terminated, env_info = self.env.step(actions[0])
            distributed_rewards = [0] * self.n_agents
            for receiver in range(self.n_agents):
                for giver in range(self.n_agents):
                    w = p[0][receiver][giver]*q[0][giver][receiver]
                    w = w.detach().numpy()
                    distributed_rewards[receiver] = rewards[giver]*math.sqrt(w)
                    # hopefully that doesn't lose too much rewards
            episode_return = [episode_return[idx] + rewards[idx] for idx in range(self.n_agents)]

            post_transition_data = {
                "actions": actions,
                "rewards": rewards,
                "redistributed_rewards": distributed_rewards,
                "terminated": [(terminated,)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        if self.env.is_masked():
            last_data = {
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
                "adjacent_agents": [self.env.get_adj()],
                "p": p,
                "q": q,
            }
        else:
            last_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
                "p": p,
                "q": q,
            }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions,}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        if sample_mode:
            return episode_return
        else:
            return self.batch


    def _log(self, returns, stats, prefix):
        for agent_idx in range(self.n_agents):
            agent_returns = [returns[t][agent_idx] for t in range(len(returns))]
            self.logger.log_stat(prefix + "agent {} return_mean".format(agent_idx), np.mean(agent_returns), self.t_env)
            self.logger.log_stat(prefix + "agent {} return_std".format(agent_idx), np.std(agent_returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
