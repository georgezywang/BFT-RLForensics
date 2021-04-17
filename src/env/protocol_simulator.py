import copy
from random import random

from env.multiagentenv import MultiAgentEnv
from protocols.PBFT.log import ClientLog
from protocols.PBFT.message import create_message
from protocols.PBFT.replica import PBFTagent

# player 0: attacker
# player 1: identifier

client_vals = [0, 1, 2, 3]
consensus_status = {"violated": 0,
                    "normal": 1}
type_dict = {"PrePrepare": 1,
             "Prepare": 2,
             "Commit": 3,
             "ViewChange": 4,
             "NewView": 5,
             "PrepareCertificate": 6,
             "CommitCertificate": 7,
             "Client": 8,
             "BlockCommit": 9,
             "RequestClient": 10, }


class ProtocolSimulator(MultiAgentEnv):
    def __init__(self, args):
        self.args = args
        self.episode_limit = args.episode_limit
        self.n_replicas = args.n_peers
        self.n_malicious = args.num_malicious
        self.max_messages_per_round = args.max_message_num_per_round
        self.honest_replicas = [PBFTagent(args) for _ in range(self.n_replicas - self.n_malicious)]

        self.honest_ids = []
        self.malicious_ids = []
        self.round_counter = 0
        self.client_request_vals = None
        self.client_request_seq_num = 0
        self.replica_msg_buffers = {}
        self.total_msgs_per_round = None
        self.replica_reply_log = None

        self.attacker_reward = 0
        self.identifier_reward = 0

    def reset(self):
        self.attacker_reward = 0
        self.identifier_reward = 0

        # assign ids
        replica_ids = random.shuffle([idx for idx in range(self.n_replicas)])
        self.malicious_ids = replica_ids[0:self.n_malicious]
        self.honest_ids = replica_ids[self.n_malicious:]
        for idx in range(self.n_replicas - self.n_malicious):
            self.honest_replicas[idx].reset(self.honest_ids[idx])
            self.replica_msg_buffers[idx] = []

        # reset basics
        self.round_counter = 0
        self.client_request_vals = {}
        self.client_request_seq_num = self.args.initialized_seq_num - 1
        self.total_msgs_per_round = []
        self.replica_reply_log = ClientLog(self.args)

        # create first client request
        self._create_new_client_request()
        # distribute to all
        self._send_messages_to_replica_buffers()

    def _create_new_client_request(self):
        self.client_request_seq_num += 1
        self.client_request_vals[self.client_request_seq_num] = random.choice(client_vals)
        params = {"msg_type": "Client",
                  "view_num": float("inf"),
                  "seq_num": self.client_request_seq_num,
                  "signer_id": self.args.simulator_id,
                  "val": self.client_request_vals[self.client_request_seq_num]}
        for r_id in range(self.n_replicas):  # no matter of primary
            t_params = copy.deepcopy(params)
            t_params["receiver_id"] = r_id
            self.total_msgs_per_round.append(create_message(self.args, t_params))

    def _send_messages_to_replica_buffers(self):
        # give out all received messages
        for msg in self.total_msgs_per_round:
            self.replica_msg_buffers[msg.receiver_id].append(copy.deepcopy(msg))
        self.total_msgs_per_round = []

        # gather responses and empty buffers of scripted protocol agents
        # optional: simulate traffic overflow scenario (fix the size of identifier's obs)
        for r in self.honest_replicas:
            self.total_msgs_per_round.extend(
                r.handle_msgs(self.replica_msg_buffers[r.id][0:self.max_messages_per_round]))
            self.replica_msg_buffers[r.id] = self.replica_msg_buffers[r.id][self.max_messages_per_round:]

    def step(self, actions):
        # act
        # update basic metrics
        # parse attacker's actions to messages
        # parse identifier's actions t
        # obtain honest replicas' messages
        # check if consensus are reached, -> rewards
        # new client request
        # send and handle messages
        self.round_counter += 1
        self.attacker_reward = 0
        self.identifier_reward = 0
        terminated = False

        attacker_messages, identifier_choices = self._parse_actions(actions)
        self.total_msgs_per_round.append(attacker_messages)

        self._handle_reply_msgs_to_client()
        status = self._check_consensus()

        if self.round_counter % self.args.steps_per_new_client_request == 0:
            self._create_new_client_request()

        self._check_identifier_correctness(identifier_choices)
        rewards = [self.attacker_reward, self.identifier_reward]

        if self.args.terminate_after_consensus_breached:
            terminated = status == consensus_status["violated"] or self.round_counter > self.args.episode_limit

        env_info = {}

        return rewards, terminated, env_info

    def _check_identifier_correctness(self, choices):
        correct = sum([1 for x in choices if x in self.malicious_ids])
        incorrect = len(choices) - len(choices)
        self.attacker_reward = correct * self.args.attacker_reward_correct_detection + incorrect * self.args.attacker_reward_incorrect_detection
        self.identifier_reward = correct * self.args.identifier_reward_correct_detection + incorrect * self.args.identifier_reward_incorrect_detection

    def _handle_reply_msgs_to_client(self):
        for msg in self.total_msgs_per_round:
            if msg.msg_type == type_dict["BlockCommit"]:
                self.replica_reply_log.get_entry(msg.seq_num).add_message(msg)

        self.total_msgs_per_round = [msg for msg in self.total_msgs_per_round if
                                     not msg.msg_type == type_dict["BlockCommit"]]

    def _check_consensus(self):
        status = consensus_status["normal"]
        for seq_num in self.replica_reply_log.entries.keys():
            if not self.replica_reply_log.get_entry(seq_num).visited:
                committed_vals = self.replica_reply_log.get_entry(seq_num).get_block_committed_val()
                if len(committed_vals) == 0:
                    continue
                if len(committed_vals) == 1:
                    if committed_vals[0] == self.client_request_vals[seq_num]:
                        self.attacker_reward += self.args.attacker_reward_consensus_reached
                        self.identifier_reward += self.args.identifier_reward_consensus_reached
                else:
                    self.attacker_reward += self.args.attacker_reward_consensus_breached
                    self.identifier_reward += self.args.identifier_reward_consensus_breached
                    status = consensus_status["violated"]
        return status

    def _parse_actions(self, actions):
        # TODO: output attacker's messages and identifier's choices of attackers
        return [], []

    def get_obs(self):
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"obs_shape": self.get_obs_size(),
                    "reward_shape": 2,
                    "n_actions": self.get_total_actions(),
                    "adjacent_agents_shape": 0,
                    "n_agents": 2,
                    "episode_limit": self.episode_limit}
        return env_info
