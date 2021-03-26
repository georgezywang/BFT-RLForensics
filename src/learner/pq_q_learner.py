import copy

import torch

from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import Adam

from module.critics.pq_critic import PQCritic


class BaiscPQ_QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.pq_critic = PQCritic(scheme, args)

        self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        self.pq_optimiser = Adam(params=list(mac.pq_parameters())+list(self.pq_critic.parameters()),
                                 lr=args.lr,
                                 eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.pq_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

    def pq_train(self, batch, running_log=None):
        p = batch["p"]  # [batch_size, n_agent, n_agent]
        q = batch["q"]  # [batch_size, n_agent, n_agent]

        # train pq critic
        pq_vals = self.pq_critic(batch)
        pq_targets = batch["evals"]
        pq_critic_loss = pq_vals - pq_targets
        pq_critic_loss = (pq_critic_loss ** 2).sum()
        running_log["control_critic_loss"].append(pq_critic_loss.item())

        # train pq actor
        log_p = torch.log(p)
        log_q = torch.log(q)
        p_loss = (log_p * pq_vals.detach()).sum()
        q_loss = (log_q * pq_vals.detach()).sum()

        loss = pq_critic_loss + p_loss + q_loss
        self.pq_optimiser.zero_grad()
        loss.backward()
        self.pq_optimiser.step()
        self.pq_training_steps += 1

        self.logger.log_stat("p_loss", p_loss.item())
        self.logger.log_stat("q_loss", q_loss.item())

    def action_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["rewards"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # self.mac.init_latent(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)  # (bs,n,n_actions)
            mac_out.append(agent_outs)  # [t,(bs,n,n_actions)]
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # (bs,t,n,n_actions), Q values of n_actions

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # (bs,t,n) Q value of an action

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)  # (bs,n,hidden_size)
        # self.target_mac.init_latent(batch.batch_size)

        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)  # (bs,n,n_actions)
            target_mac_out.append(target_agent_outs)  # [t,(bs,n,n_actions)]

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time, dim=1 is time index
        # (bs,t,n,n_actions)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # Q values


        target_max_qvals = target_mac_out.max(dim=3)[0]

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())  # no gradient through target net
        # (bs,t,1)

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data (LSE)
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)  # max_norm
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.pq_critic.cuda()
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        self.pq_critic.save_models(path)
        th.save(self.pq_optimiser.state_dict(), "{}/pq_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.pq_critic.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/pq_opt.th".format(path), map_location=lambda storage, loc: storage))
