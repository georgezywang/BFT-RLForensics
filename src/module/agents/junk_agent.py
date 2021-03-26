from torch import nn

from module.utils.components import MLP


class PQAgent(nn.Module):
    """
    capable of reasoning its learning?? - how to do it in a sampling efficient way
    [It is an agent that is probably not going to score really well on its task(s), but you know what, it will lead to
    better agents in the future.]
    pq selection:
        actor-critic framework
        input: (its own pq, other pq) (+ minimax)
    action selection: (recurrent)
        input: (obs, current pq)
    """

    def __init__(self, args, scheme):
        super(PQAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        pq_input_shape = self.n_agents
        pq_output_shape = self.n_agents
        self.actor = MLP(self.args.pq_actor_hidden_sizes, pq_input_shape, pq_output_shape, args)

    def forward(self, batch):
        return self.actor(batch)




