import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import Adam

from module.critics.dist_critic import DistCritic


class MetaQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.z_critic = DistCritic(scheme, args)

        self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        self.z_learning_optimiser = Adam(params=list(self.z_critic.parameters()), lr=args.z_critic_lr, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def get_social_welfare_z(self, batch, device):
        z_vals = self.z_critic(batch, device)
        return z_vals.sum() / self.args.n_agents

    def z_train(self, batch, device, t_env):
        bs = batch["evals"].shape[0]

        z_vals = self.z_critic(batch, device)
        z_targets = batch["evals"]
        z_critic_loss = z_vals - z_targets
        z_critic_loss = (z_critic_loss ** 2).sum()/bs
        self.logger.log_stat("z_critic_loss", z_critic_loss.item(), t_env)

        self.z_learning_optimiser.zero_grad()
        z_critic_loss.backward()
        self.z_learning_optimiser.step()

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
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

        kl_divs = []
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)  # (bs,n,n_actions)
            kl_divs.append(self.mac.compute_kl_div())
            mac_out.append(agent_outs)  # [t,(bs,n,n_actions)]
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        kl_divs = th.stack(kl_divs, dim=1)[:, :-1]
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

        kl_mask = copy.deepcopy(mask).expand_as(kl_divs)
        masked_kl_div = kl_divs * kl_mask
        kl_div_loss = (masked_kl_div ** 2).sum() / kl_mask.sum()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data (LSE)
        td_error_loss = (masked_td_error ** 2).sum() / mask.sum()

        loss = td_error_loss + kl_div_loss

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
            self.logger.log_stat("kl_div_abs", kl_div_loss.abs().item(), t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
