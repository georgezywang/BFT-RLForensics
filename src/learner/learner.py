import copy

from components.critic.critic import Critic
from components.episode_buffer import EpisodeBatch
import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from utils.rl_utils import build_td_lambda_targets


class SeparateLearner:
    def __init__(self, mac, scheme, logger, args):
        self.n_peers = args.n_peers
        self.n_agents = args.n_agents
        self.args = args
        self.device = th.device("cuda:0" if th.cuda.is_available() and self.args.use_cuda else "cpu")
        self.mac = mac
        self.logger = logger

        self.critic = Critic(scheme, args)

        self.last_target_update_step = 0
        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.identifier_params = list(mac.identifier_parameters())
        self.attacker_params = list(mac.attacker_parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.identifier_params + self.attacker_params + self.critic_params

        self.identifier_optimiser = Adam(params=self.identifier_params, lr=args.lr, eps=args.optim_eps)
        self.attacker_optimiser = Adam(params=self.attacker_params, lr=args.lr, eps=args.optim_eps)
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.target_critic = Critic(scheme, args)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        b_len = batch.max_seq_length
        identifier_rewards = batch["identifier_reward"][:, :-1]
        attacker_rewards = batch["attacker_reward"][:, :-1]
        identifier_actions = batch["identifier_action"][:, :-1]
        attacker_actions = batch["attacker_action"][:, :-1]
        # print(attacker_actions)
        attacker_actions = self._parse_attacker_actions(attacker_actions)
        # print("done with action parsing")
        # print(attacker_actions)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()  # [bs, t-1, 1]
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        critic_mask = mask.clone().repeat(1, 1, self.n_agents)

        rewards = (attacker_rewards, identifier_rewards)
        q_vals, critic_train_stats = self._train_critic(batch, rewards, terminated, critic_mask)  # [bs, t, 2]
        # print("done with critic")

        # Calculate estimated Q-Values
        attacker_outs = []
        identifier_outs = []
        self.mac.init_hidden(bs)
        for t in range(batch.max_seq_length):
            attacker_out, identifier_out = self.mac.forward(batch, t=t, test_mode=False)  # (bs,n,n_actions)
            attacker_outs.append(attacker_out)  # [t,(bs,n_peers)]
            identifier_outs.append(identifier_out)

        # learn identifier actor
        identifier_outs = th.stack(identifier_outs, dim=1).unsqueeze(-1)  # Concat over time
        # print("identifier_outs_shape: {}".format(identifier_outs.shape))
        # print("identifier_outs: {}".format(identifier_outs))
        identifier_outs = th.cat([identifier_outs, 1 - identifier_outs], dim=-1).reshape(bs, b_len, self.n_peers, 2)
        identifier_outs = F.softmax(identifier_outs, dim=-1)  # TODO: This is bad
        # print("identifier_outs_shape: {}".format(identifier_outs.shape))
        # print("stacked_identifier_outs: {}".format(identifier_outs[0][0]))
        # (bs,t,n,n_actions), Q values of n_actions

        # Pick the Q-Values for the actions taken by each agent
        # print(identifier_actions)
        identifier_actions = identifier_actions.unsqueeze(3).type(th.int64)
        # print("identifier_actions_shape: {}".format(identifier_actions.shape))
        # print("identifier_actions: {}".format(identifier_actions[0][0]))
        identifier_chosen_action_pi = th.gather(identifier_outs[:, :-1], dim=3, index=identifier_actions).squeeze(
            3)  # Remove the last dim
        # print("identifier_chosen_action_pi: {}".format(identifier_chosen_action_pi[0][0]))
        identifier_mask = mask.clone().repeat(1, 1, self.n_peers)
        # print("identifier_mask: {}".format(identifier_mask[0][0]))
        identifier_chosen_action_pi[identifier_mask == 0] = 1.0
        # print("identifier_chosen_action_pi after mask: {}".format(identifier_chosen_action_pi[0][1]))
        log_identifier_pi = th.log(identifier_chosen_action_pi).sum(dim=-1)
        # print("log_identifier_pi: {}".format(log_identifier_pi[0][0]))

        identifier_critic = mask.clone().reshape(-1)
        identifier_loss = ((q_vals[:, :, 1].reshape(-1).detach() * log_identifier_pi.reshape(
            -1)) * identifier_critic).sum() / identifier_critic.sum()
        # print("q_vals 001: {}".format(q_vals[0][0][1]))
        # print("log_identifier_pi: {}".format(log_identifier_pi[0][0]))
        # print("q_vals * log_identifier_pi: {}".format((q_vals[:, :, 1].reshape(-1) * log_identifier_pi.reshape(-1))[0]))
        # print("q_vals 011: {}".format(q_vals[0][1][1]))
        # print("log_identifier_pi: {}".format(log_identifier_pi[0][1]))
        # print("q_vals * log_identifier_pi: {}".format((q_vals[:, :, 1].reshape(-1) * log_identifier_pi.reshape(-1))[1]))
        # print(q_vals[:, :, 1].reshape(-1) * log_identifier_pi.reshape(-1))
        # print("mask shape: {}".format(mask.shape))
        self.identifier_optimiser.zero_grad()
        identifier_loss.backward()
        identifier_grad_norm = th.nn.utils.clip_grad_norm_(self.identifier_params, self.args.grad_norm_clip)
        self.identifier_optimiser.step()
        # print("done with identifier")
        num_action_types = len(attacker_outs[0])
        total_msgs_num = self.args.num_malicious * self.args.max_message_num_per_round
        attacker_mask = mask.clone().repeat(1, 1, total_msgs_num)
        pi = []
        for idx in range(num_action_types - 1):
            out = th.stack([attacker_outs[t][idx] for t in range(len(attacker_outs))],
                           dim=1)  # [bs, t, max_msg, num_action]
            # print(out.shape)
            # print(attacker_actions[idx])
            # print(out.is_cuda)
            # print(attacker_actions[idx].is_cuda)
            attacker_action = attacker_actions[idx].unsqueeze(3)
            out = th.gather(out[:, :-1], dim=3, index=attacker_action).squeeze(3)
            out[attacker_mask == 0] = 1.0
            pi.append(out)
        pi = th.cat(pi, dim=-1)

        cert_pi = []
        for r_id in range(self.n_peers):  # ([bs, max_msg_num, 2])*n_peers
            out = th.stack([attacker_outs[t][-1][r_id] for t in range(len(attacker_outs))],
                           dim=1)  # [bs, t, max_msg, 2]
            attacker_action = attacker_actions[-1][r_id].unsqueeze(3)
            out = th.gather(out[:, :-1], dim=3, index=attacker_action).squeeze(3)
            out[attacker_mask == 0] = 1.0
            cert_pi.append(out)
        cert_pi = th.cat(cert_pi, dim=-1)
        pi = th.cat([pi, cert_pi], dim=-1)
        log_attacker_pi = th.log(pi).sum(-1)

        attacker_critic = mask.clone().reshape(-1)
        attacker_loss = ((q_vals[:, :, 0].reshape(-1).detach() * log_attacker_pi.reshape(
            -1)) * attacker_critic).sum() / attacker_critic.sum()
        self.attacker_optimiser.zero_grad()
        attacker_loss.backward()
        attacker_grad_norm = th.nn.utils.clip_grad_norm_(self.attacker_params, self.args.grad_norm_clip)
        self.attacker_optimiser.step()
        # print("done with attacker")

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat("identifier_actor_loss", identifier_loss.item(), t_env)
            self.logger.log_stat("identifier_grad_norm", identifier_grad_norm, t_env)
            self.logger.log_stat("attacker_actor_loss", attacker_loss.item(), t_env)
            self.logger.log_stat("attacker_grad_norm", attacker_grad_norm, t_env)
            self.log_stats_t = t_env

    def _train_critic(self, batch, rewards, terminated, mask):
        # Optimise critic
        rewards = th.cat(rewards, dim=-1).detach()  # [bs, t, 2]

        target_critic_outs = []
        target_critic_hidden = self.target_critic.init_hidden().expand(batch.batch_size, -1)
        for t in range(batch.max_seq_length):
            out, target_critic_hidden = self.target_critic.forward(batch, target_critic_hidden, t)  # (bs, 2)
            target_critic_outs.append(out)  # [t,(bs, 2)]
        target_critic_outs = th.stack(target_critic_outs, dim=1)  # [bs, t, 2]

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, target_critic_outs, self.n_agents, self.args.gamma,
                                          self.args.td_lambda).detach()
        # print("target shape： {}".format(targets.shape))

        q_vals = th.zeros_like(target_critic_outs)[:, :-1]  # [bs, t-1, 2]
        critic_hidden = self.critic.init_hidden().expand(batch.batch_size, -1)
        for t in reversed(range(rewards.size(1))):
            q_t, critic_hidden = self.critic(batch, critic_hidden, t)
            q_vals[:, t] = q_t

        td_error = (q_vals - targets)
        mask_t = mask.clone().expand(-1, -1, self.n_agents)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask_t

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask_t.sum()
        self.critic_optimiser.zero_grad()
        loss.backward()

        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.critic_training_steps += 1

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm)
        mask_elems = mask_t.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_taken_mean"].append((q_vals * mask_t).sum().item() / mask_elems)
        running_log["target_mean"].append((targets * mask_t).sum().item() / mask_elems)

        return q_vals, running_log

    def _parse_attacker_actions(self, actions):
        # actions: [bs, t, num_max_msgs * ]
        bs = actions.size(0)
        t_len = actions.size(1)
        total_msgs_num = self.args.num_malicious * self.args.max_message_num_per_round
        # print("actions_shape: {}".format(actions.shape))
        actions = actions.reshape(bs, t_len, total_msgs_num, -1)
        parsed_actions = []
        for bs_idx in range(bs):
            parsed_actions_t = []
            for t_idx in range(t_len):
                parsed_actions_t_msg = []
                for msg_idx in range(total_msgs_num):
                    parsed_actions_t_msg.append(self._parse_input_message(actions[bs_idx][t_idx][msg_idx]).unsqueeze(0))
                parsed_actions_t.append(th.cat(parsed_actions_t_msg, dim=0).unsqueeze(0))
            parsed_actions.append(th.cat(parsed_actions_t, dim=0).unsqueeze(0))

        parsed_actions = th.cat(parsed_actions, dim=0)
        print(parsed_actions.shape)
        num_msg_type = 10
        ret = list(th.split(parsed_actions, [num_msg_type,
                                             self.args.num_malicious,
                                             self.args.max_view_num,
                                             self.args.max_seq_num,
                                             self.args.total_client_vals,
                                             self.args.n_peers,
                                             self.args.n_peers * 2], dim=-1))
        ret_cert = list(ret[-1].split(2, dim=-1))
        ret = list(ret[:-1])
        ret.append(ret_cert)
        return ret

    def _parse_input_message(self, msg_input):
        num_msg_type = 10  # no client type, 9 is no-op
        # msg_action_space = num_msg_type + self.args.num_malicious + \
        #                    self.args.episode_limit / 4 + self.args.episode_limit / 4 + \
        #                    len(client_vals) + self.args.n_peers + self.args.n_peers

        msg = []
        idx = 0

        msg_type_input = msg_input[idx: num_msg_type]
        msg.append(rev_onehot(msg_type_input))
        # print("msg input shape: {}".format(msg_input.shape))
        # print("msg type: {}".format(msg_type_input))
        # print("msg type: {}".format(rev_onehot(msg_type_input)))

        idx = num_msg_type
        sender_id_input = msg_input[idx: idx + self.args.num_malicious]

        msg.append(rev_onehot(sender_id_input))
        # print("sender id shape: {}".format(sender_id_input.shape))
        # print("sender id: {}".format(sender_id_input))
        # print("sender id: {}".format(rev_onehot(sender_id_input)))

        idx += self.args.num_malicious
        view_num_input = msg_input[idx: idx + self.args.max_view_num]
        msg.append(rev_onehot(view_num_input))

        idx += self.args.max_view_num
        seq_num_input = msg_input[idx: idx + self.args.max_seq_num]
        msg.append(rev_onehot(seq_num_input))

        idx += self.args.max_seq_num
        val_input = msg_input[idx: idx + self.args.total_client_vals]
        msg.append(rev_onehot(val_input))

        idx += self.args.total_client_vals
        receiver_id_input = msg_input[idx: idx + self.args.n_peers]
        msg.append(rev_onehot(receiver_id_input))

        idx += self.args.n_peers
        certificate_input = msg_input[idx:]
        msg.append(list_rev_onehot(certificate_input))
        # print("certificate shape: {}".format(certificate_input.shape))
        # print("c id: {}".format(certificate_input))
        # print("c id: {}".format(list_rev_onehot(certificate_input)))
        # print("msg: {}".format(msg))
        msg = th.cat(msg, dim=-1)

        return msg

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.target_critic.state_dict(), "{}/tar_critic.th".format(path))
        th.save(self.identifier_optimiser.state_dict(), "{}/identifier_critic_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        th.save(self.attacker_optimiser.state_dict(), "{}/attacker_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(
            th.load("{}/tar_critic.th".format(path), map_location=lambda storage, loc: storage))
        self.identifier_optimiser.load_state_dict(
            th.load("{}/identifier_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.attacker_optimiser.load_state_dict(
            th.load("{}/attacker_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))


def list_rev_onehot(x):  # for certificates
    return x.reshape(-1, 2).argmax(dim=-1)


def rev_onehot(x):  # anyvalue if invalid, will be masked out
    # print(x)
    return x.argmax().unsqueeze(0)
