"""
Code adapted from https://github.com/TonghanWang/ROMA
"""

from functools import partial

import numpy as np

from components.episode_buffer import EpisodeBatch
from env import REGISTRY as env_REGISTRY


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](self.args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.identifier_train_returns = []
        self.attacker_train_returns = []
        self.identifier_test_returns = []
        self.attacker_test_returns = []
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

    def run(self, test_mode=False):  # run one eps
        self.reset()

        terminated = False
        attacker_episode_return = 0
        identifier_episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "attacker_obs": [self.env.get_attacker_obs()],
                "identifier_obs": [self.env.get_identifier_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            attacker_actions, identifier_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # pq: [1, agent_num, control_dim]

            rewards, terminated, env_info = self.env.step(attacker_actions[0], identifier_actions[0])

            attacker_episode_return += rewards[0]
            identifier_episode_return += rewards[1]

            post_transition_data = {
                "attacker_action": attacker_actions,
                "identifier_action": identifier_actions,
                "attacker_reward": rewards[0],
                "identifier_reward": rewards[1],
                "terminated": [(terminated,)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "attacker_obs": [self.env.get_attacker_obs()],
            "identifier_obs": [self.env.get_identifier_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        attacker_actions, identifier_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"attacker_action": attacker_actions,
                           "identifier_action": identifier_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        attacker_cur_returns = self.attacker_test_returns if test_mode else self.attacker_train_returns
        identifier_cur_returns = self.identifier_test_returns if test_mode else self.identifier_train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        attacker_cur_returns.append(attacker_episode_return)
        identifier_cur_returns.append(identifier_episode_return)

        if test_mode and (len(self.attacker_test_returns) == self.args.test_nepisode):
            self._log(attacker_cur_returns, identifier_cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(attacker_cur_returns, identifier_cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, attacker_returns, identifier_returns, stats, prefix):

        self.logger.log_stat(prefix + "attacker agent return_mean {} and std {}".format(np.mean(attacker_returns),
                                                                                        np.std(attacker_returns),
                                                                                        self.t_env))
        attacker_returns.clear()

        self.logger.log_stat(prefix + "identifier agent return_mean {} and std {}".format(np.mean(identifier_returns),
                                                                                          np.std(identifier_returns),
                                                                                          self.t_env))
        identifier_returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
