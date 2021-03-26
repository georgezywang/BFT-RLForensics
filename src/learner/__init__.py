from learner.nash_q_learner import NashQLearner
from learner.pq_q_learner import BaiscPQ_QLearner
from learner.q_learner import QLearner

REGISTRY = {}

REGISTRY["nash_q_learner"] = NashQLearner
REGISTRY["q_learner"] = QLearner
REGISTRY["pq_q_learner"] = BaiscPQ_QLearner
