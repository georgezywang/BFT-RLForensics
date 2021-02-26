import copy
from components.episode_buffer import EpisodeBatch
import torch
from torch.optim import Adam


class NashQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["rewards"][:, :-1]
        redistributed_rewards = batch["redistributed_rewards"][:, :-1]
        actions = batch["actions"][:, :-1]
        pq = batch["pq"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_step_out = []
        mac_pq_vals_out = []
        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length-1):
            step_out, _, _, control_state_out = self.mac.forward(batch, t=t)  # (bs,n,n_actions)
            pq_vals_out = self.mac.index_to_pq_vals(pq[:, t], control_state_out)
            mac_step_out.append(step_out)  # [t,(bs,n,n_actions)]
            mac_pq_vals_out.append(pq_vals_out)  # [t, (bs, n)]
        mac_step_out = torch.stack(mac_step_out, dim=1)  # Concat over time
        chosen_qp_vals = torch.stack(mac_pq_vals_out, dim=1)
        # (bs,t,n,n_actions), Q values of n_actions, ()

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = torch.gather(mac_step_out, dim=3, index=actions).squeeze(3)  # Remove the last dim
        # (bs,t,n) Q value of an action

        step_error = (chosen_action_qvals - rewards)
        step_mask = mask.clone().expand_as(step_error)
        masked_step_error = step_error * step_mask
        step_loss = (masked_step_error ** 2).sum() / step_mask.sum()

        # Calculate the Q-Values necessary for the target
        target_mac_ne_out = []
        self.target_mac.init_hidden(batch.batch_size)  # (bs,n,hidden_size)

        for t in range(batch.max_seq_length):
            target_step_out, _, _, target_control_out = self.target_mac.forward(batch, t=t)
            target_ne_vals_out = self.mac.compute_equlibrium(raw_control_values=target_control_out,
                                                             raw_step_values=target_step_out)
            target_mac_ne_out.append(target_ne_vals_out)  # [t,(bs,n,1)]

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_ne_vals = torch.stack(target_mac_ne_out[1:], dim=1)  # Concat across time, dim=1 is time index
        # (bs,t,n,1)

        # Calculate 1-step Q-Learning targets
        pq_estimate_targets = redistributed_rewards + self.args.gamma * (1 - terminated) * target_ne_vals

        # Td-error
        td_error = (chosen_qp_vals - pq_estimate_targets.detach())  # no gradient through target net
        # (bs,t,n, 1)

        qp_mask = mask.clone().expand_as(td_error)

        # 0-out the targets that came from padded data
        qp_masked_td_error = td_error * qp_mask

        # Normal L2 loss, take mean over actual data (LSE)
        qp_loss = (qp_masked_td_error ** 2).sum() / qp_mask.sum()

        loss = step_loss + qp_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)  # max_norm
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            qp_mask_elems = qp_mask.sum().item()
            self.logger.log_stat("qp_td_error_abs", (qp_masked_td_error.abs().sum().item() / qp_mask_elems), t_env)
            step_mask_elems = step_mask.sum().item()
            self.logger.log_stat("step_error_abs", (masked_step_error.abs().sum().item() / step_mask_elems), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser.load_state_dict(torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
