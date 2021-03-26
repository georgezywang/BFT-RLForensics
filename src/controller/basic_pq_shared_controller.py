import torch
import torch as th

from components.action_selectors import EpsilonGreedyActionSelector
from controller.shared_controller import BasicMAC
from module.agents import REGISTRY as agent_REGISTRY


# This multi-agent controller shares parameters between agents
class BasicPQMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(BasicPQMAC, self).__init__(scheme=scheme, groups=groups, args=args)
        self.n_agents = args.n_agents
        self.args = args
        self.scheme = scheme
        self._build_agents()
        self._build_pq_actors()
        self.agent_output_type = args.agent_output_type
        self.action_selector = EpsilonGreedyActionSelector(args)
        self.hidden_states = None

    def select_pqs(self, batch_size, device, eps_num, test_mode):
        p_outs, q_outs = self.pq_forward(batch_size, device, test_mode)
        # agent actions
        if self.args.pq_output_type == "pi_logits":

            p_outs = torch.nn.functional.softmax(p_outs, dim=-1)
            q_outs = torch.nn.functional.softmax(q_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = p_outs.size(-1)
                # FIXME: own exploration probability
                # With probability epsilon, we will pick an available action uniformly
                p_outs = ((1 - self.action_selector.epsilon) * p_outs
                          + torch.ones_like(p_outs) * self.action_selector.epsilon / epsilon_action_num)
                q_outs = ((1 - self.action_selector.epsilon) * q_outs
                          + torch.ones_like(q_outs) * self.action_selector.epsilon / epsilon_action_num)

        p, q = self.action_selector.select_individuak_pq_values(p_outs, q_outs, eps_num,
                                                                    test_mode=test_mode)
        return self._one_hot_embedding(p, self.args.n_agents), self._one_hot_embedding(q, self.args.n_agents)

    def pq_forward(self, batch_size, device, test_mode=False):
        # inputs: actor_id
        p_inputs, q_inputs = self._build_pq_inputs(batch_size, device)
        p_outs = self.p_actor(p_inputs)
        q_outs = self.q_actor(q_inputs)
        return p_outs, q_outs

    def pq_parameters(self):
        return list(self.p_actor.parameters()) + list(self.q_actor.parameters())

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.p_actor.load_state_dict(other_mac.p_actor.state_dict())
        self.q_actor.load_state_dict(other_mac.q_actor.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.p_actor.cuda()
        self.q_actor.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.p_actor.state_dict(), "{}/p_actor.th".format(path))
        th.save(self.q_actor.state_dict(), "{}/q_actor.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.agent.load_state_dict(th.load("{}/p_actor.th".format(path), map_location=lambda storage, loc: storage))
        self.agent.load_state_dict(th.load("{}/q_actor.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self):
        self.agent = agent_REGISTRY[self.args.agent](self.args, self.scheme)

    def _build_pq_actors(self):
        self.p_actor = agent_REGISTRY[self.args.p_actor](self.args, self.scheme)
        self.q_actor = agent_REGISTRY[self.args.q_actor](self.args, self.scheme)

    def _build_pq_inputs(self, batch_size, device):
        bs = batch_size
        p_inputs = [th.eye(self.n_agents, device=device).unsqueeze(0).expand(bs, -1, -1)]
        p_inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in p_inputs], dim=1)
        q_inputs = [th.eye(self.n_agents, device=device).unsqueeze(0).expand(bs, -1, -1)]
        q_inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in q_inputs], dim=1)
        return p_inputs, q_inputs

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs.append(batch["q"][:, t - 1].reshape(bs, -1).unsqueeze(1).expand(-1, self.n_agents, -1))
        inputs.append(batch["p"][:, t - 1].reshape(bs, -1).unsqueeze(1).expand(-1, self.n_agents, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _one_hot_embedding(self, labels, num_classes):
        """Embedding labels to one-hot form.

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N, #classes].
        """
        y = torch.eye(num_classes)
        return y[labels]


