import torch as th

from components.action_selectors import EpsilonGreedyActionSelector
from module.agents import REGISTRY as agent_REGISTRY


# This multi-agent controller shares parameters between agents
from module.utils import MLPMultiGaussianEncoder


class BasicLatentMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.scheme = scheme
        self._build_agents()
        self.agent_output_type = args.agent_output_type
        self.action_selector = EpsilonGreedyActionSelector(args)
        latent_input_shape, latent_output_shape, latent_hidden_sizes = self._get_latent_shapes()
        self.latent_encoder = MLPMultiGaussianEncoder(latent_input_shape, latent_output_shape, latent_hidden_sizes)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def sample_batch_latent_var(self, batch, t):
        # z_q, z_p: [n_agents][latent_relation_space_dim]
        inputs = self._build_latent_encoder_input(batch, t)
        latent_var = self.latent_encoder.infer_posterior(inputs)
        if len(latent_var.shape) == 1:
            latent_var = latent_var.unsqueeze(0)
        return latent_var

    # def sample_latent_var(self, z_q, z_p):
    #     inputs = th.cat([x.reshape(-1) for x in [z_q, z_p]], dim=1)
    #     return self.latent_encoder.infer_posterior(inputs)

    def compute_kl_div(self):
        div = self.latent_encoder.compute_kl_div()
        self.latent_encoder.reset()
        return div

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        mask = ep_batch["adjacent_agents"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, mask, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":  # (0, 1) -> (-inf, inf)

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        if 'init_hidden' in self.agent.__dict__:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return list(self.agent.parameters())+list(self.latent_encoder.parameters())

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.latent_encoder.load_state_dict(other_mac.latent_encoder.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.latent_encoder.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.latent_encoder.state_dict(), "{}/encoder.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.latent_encoder.load_state_dict(th.load("{}/encoder.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self):
        self.agent = agent_REGISTRY[self.args.agent](self.args, self.scheme)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs.append(self.sample_batch_latent_var(batch, t).unsqueeze(1).expand(-1, self.n_agents, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_latent_shapes(self):
        # input shape: z, z' FIXME: could also use parsed relationship as input
        latent_input_shape = self.args.latent_relation_space_dim*self.n_agents*2
        latent_output_shape = self.args.latent_var_dim
        latent_hidden_sizes = self.args.latent_encoder_hidden_sizes
        return latent_input_shape, latent_output_shape, latent_hidden_sizes

    def _build_latent_encoder_input(self, batch, t):
        # z_q, z_p: [n_agents][latent_relation_space_dim]
        bs = batch.batch_size
        if len(batch["z_q"][:, t].shape) == 1:  # FIXME: probably unnecessary
            inputs = [batch["z_q"][:, t].unsqueeze(0), batch["z_p"][:, t].unsqueeze(0)]
        else:
            inputs = [batch["z_q"][:, t], batch["z_p"][:, t]]
        return th.cat([x.reshape(bs, -1) for x in inputs], dim=1)


