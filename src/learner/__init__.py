from learner.nash_q_learner import NashQLearner
from learner.q_learner import QLearner

REGISTRY = {}

REGISTRY["nash_q_learner"] = NashQLearner
REGISTRY["q_learner"] = QLearner