from module.agents.dgn_agent import DGNAgent
from module.agents.junk_agent import PQAgent
from module.agents.nash_q_agent import NashQAgent
from module.agents.rnn_agent import RNNAgent

REGISTRY = {}

REGISTRY["nash_q_agent"] = NashQAgent
REGISTRY["rnn_agent"] = RNNAgent
REGISTRY["dgn_agent"] = DGNAgent
REGISTRY["pq_agent"] = PQAgent