import copy
import random
import numpy as np

from env.multiagentenv import MultiAgentEnv
from protocols.PBFT.log import ClientLog
from protocols.PBFT.message import create_message
from protocols.PBFT.replica import PBFTagent, PBFTagent_wrapper

# player 0: attacker
# player 1: identifier

client_vals = [0, 1, 2, 3]
consensus_status = {"violated": 0,
                    "normal": 1}
type_dict = {"PrePrepare": 0,
             "Prepare": 1,
             "Commit": 2,
             "ViewChange": 3,
             "NewView": 4,
             "PrepareCertificate": 5,
             "CommitCertificate": 6,
             "BlockCommit": 7,
             "RequestClient": 8,
             "No-op": 9,
             "Client": 10, }


class ProtocolSimulator(MultiAgentEnv):
    def __init__(self, args):
        self.args = args
        self.episode_limit = args.episode_limit
        self.n_replicas = args.n_peers
        self.n_malicious = args.num_malicious
        self.max_messages_per_round = args.max_message_num_per_round
        self.honest_replicas = [PBFTagent(args) for _ in range(self.n_replicas - self.n_malicious)]
        self.malicious_replicas = [PBFTagent_wrapper(args) for _ in range(self.n_malicious)]

        self.honest_ids = []
        self.malicious_ids = []
        self.round_counter = 0
        self.client_request_vals = None
        self.client_request_seq_num = 0
        self.replica_msg_buffers = {}
        self.replica_static_msg_buffers = {}  # only for collecting messages for obs
        self.total_msgs_per_round = None
        self.replica_reply_log = None

        self.attacker_reward = 0
        self.identifier_reward = 0

        self.identifier_transcript_ids = None

    def reset(self):
        self.attacker_reward = 0
        self.identifier_reward = 0

        # assign ids
        replica_ids = [idx for idx in range(self.n_replicas)]
        random.shuffle(replica_ids)
        self.malicious_ids = replica_ids[0:self.n_malicious]
        self.honest_ids = replica_ids[self.n_malicious:]

        # initializes honest replicas
        for idx in range(self.n_replicas - self.n_malicious):
            self.honest_replicas[idx].reset(self.honest_ids[idx])

        # initializes malicious replicas
        for idx in range(self.n_malicious):
            self.malicious_replicas[idx].reset(self.malicious_ids[idx])

        # initialize msg buffers
        for idx in range(self.n_replicas):
            self.replica_msg_buffers[idx] = []
            self.replica_static_msg_buffers[idx] = []

        # assign transcripts accessible to the identifier
        self.identifier_transcript_ids = random.sample(range(self.n_replicas), self.args.num_transcripts_avail)

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

        for r_id in range(self.n_replicas):
            self.replica_static_msg_buffers[r_id] = copy.deepcopy(
                self.replica_msg_buffers[r_id][:self.max_messages_per_round])

        # gather responses and empty buffers of scripted protocol agents
        # optional: simulate traffic overflow scenario (fix the size of identifier's obs)
        for r in self.honest_replicas:
            # print(len(self.replica_msg_buffers[r.id]))
            # print(len(self.replica_msg_buffers[r.id][0:self.max_messages_per_round]))
            response = r.handle_msgs(self.replica_msg_buffers[r.id][0:self.max_messages_per_round])
            # print(len(response))
            self.total_msgs_per_round.extend(response)
            self.replica_msg_buffers[r.id] = self.replica_msg_buffers[r.id][self.max_messages_per_round:]

        for r in self.malicious_replicas:
            r.handle_msgs(self.replica_msg_buffers[r.id][0:self.max_messages_per_round])
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

        attacker_action, identifier_action = actions
        attacker_messages, identifier_choices = self._parse_actions(attacker_action, identifier_action)
        self.total_msgs_per_round.append(attacker_messages)

        self._handle_reply_msgs_to_client()
        status = self._check_consensus()

        if self.round_counter % self.args.steps_per_new_client_request == 0:
            self._create_new_client_request()

        self._check_identifier_correctness(identifier_choices)
        rewards = [self.attacker_reward, self.identifier_reward]

        if self.args.terminate_after_consensus_breached:
            terminated = (status == consensus_status["violated"] or self.round_counter > self.args.episode_limit
                          or self.client_request_seq_num > self.args.max_seq_num)

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

    def _parse_actions(self, attacker_action, identifier_action):
        # TODO: output attacker's messages and identifier's choices of attackers
        num_msg_type = 10  # no client type, 9 is no-op
        msg_action_space = int(num_msg_type + self.args.num_malicious + \
                               self.args.max_seq_num + self.args.max_view_num + \
                               len(client_vals) + self.args.n_peers + self.args.n_peers * 2)
        attacker_action_d = np.reshape(attacker_action, (self.args.max_message_num_per_round, msg_action_space))
        attacker_ret = []
        for idx in range(self.args.max_message_num_per_round):
            msg = self._parse_input_message(attacker_action_d[idx])
            if msg is not None:
                attacker_ret.append(msg)

        identifier_ret = [r_id for r_id in range(self.n_replicas) if identifier_action[r_id] == 1]
        return attacker_ret, identifier_ret

    def get_attacker_obs(self):
        obs = []
        idx = 0
        for r_id in self.malicious_ids:
            obs.extend(onehot(idx, self.n_malicious))
            for msg in self.replica_static_msg_buffers[r_id]:
                obs.extend(self._replica_msg_to_malicious_input(msg))
            num_decoy_msg = self.max_messages_per_round - len(self.replica_static_msg_buffers[r_id])
            obs.extend(self._decoy_msgs(num_decoy_msg, malicious=True))
            idx += 1
        # print("len of attacker_obs: {}".format(len(obs)))
        return obs

    def get_identifier_obs(self):
        obs = []
        for r_id in self.identifier_transcript_ids:
            obs.extend(onehot(r_id, self.n_replicas))
            for msg in self.replica_static_msg_buffers[r_id]:
                obs.extend(self._replica_msg_to_input(msg))
            num_decoy_msg = self.max_messages_per_round - len(self.replica_static_msg_buffers[r_id])
            obs.extend(self._decoy_msgs(num_decoy_msg))
        return obs

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        return self.get_avail_actions()[agent_id]

    def get_env_info(self):
        env_info = {"identifier_obs_shape": self.get_identifier_obs_size(),
                    "attacker_obs_shape": self.get_attacker_obs_size(),
                    "identifier_reward_shape": 1,
                    "attacker_reward_shape": 1,
                    "n_identifier_actions": self.get_identifier_action_size(),
                    "n_attacker_actions": self.get_attacker_action_size(),
                    "episode_limit": self.episode_limit}
        return env_info

    def get_attacker_action_size(self):
        num_msg_type = 10  # no client type, 9 is no-op
        msg_action_space = num_msg_type + self.args.num_malicious + \
                           self.args.max_view_num + self.args.max_seq_num + \
                           len(client_vals) + self.args.n_peers + self.args.n_peers * 2
        return int(msg_action_space * self.args.max_message_num_per_round * self.args.num_malicious)

    def get_identifier_action_size(self):
        return self.args.n_peers

    def get_attacker_obs_size(self):
        num_msg_type = 11  # with client
        msg_obs_space = num_msg_type + self.args.max_view_num + self.args.max_seq_num + self.args.n_peers + \
                        len(client_vals) + self.n_malicious + self.args.n_peers * 2
        malicious_ids = self.n_malicious * self.n_malicious
        obs_size = int(self.args.max_message_num_per_round * msg_obs_space * self.args.num_malicious + malicious_ids)
        # print("attacker_obs_size: {}, msg_obs_space: {}".format(obs_size, msg_obs_space))
        return obs_size

    def get_identifier_obs_size(self):
        num_msg_type = 11  # with client
        msg_obs_space = num_msg_type + self.args.max_view_num + self.args.max_seq_num + self.args.n_peers + \
                        len(client_vals) + self.args.n_peers + self.args.n_peers * 2
        transcript_ids = self.n_replicas * self.args.num_transcripts_avail
        return int(
            self.args.max_message_num_per_round * self.args.num_transcripts_avail * msg_obs_space + transcript_ids)

    def _parse_input_message(self, msg_input):
        num_msg_type = 10
        params = {}
        idx = 0
        msg_type_input = msg_input[idx: num_msg_type]
        params["msg_type"] = rev_onehot(msg_type_input)
        if params["msg_type"] == type_dict["No-op"]:
            return None

        idx = num_msg_type
        sender_id_input = msg_input[idx: idx + self.n_malicious]
        params["signer_id"] = self.malicious_ids[rev_onehot(sender_id_input)]

        idx += self.n_malicious
        view_num_input = msg_input[idx: idx + self.args.max_view_num]
        params["view_num"] = rev_onehot(view_num_input)

        idx += self.args.max_view_num
        seq_num_input = msg_input[idx: idx + self.args.max_seq_num]
        params["seq_num"] = rev_onehot(seq_num_input)

        idx += self.args.max_seq_num
        val_input = msg_input[idx: idx + len(client_vals)]
        params["val"] = rev_onehot(val_input)

        idx += len(client_vals)
        receiver_id_input = msg_input[idx: idx + self.n_replicas]
        params["receiver_id"] = rev_onehot(receiver_id_input)

        idx += self.n_replicas
        certificate_input = msg_input[idx:]
        params["certificate"] = rev_list_onehot(certificate_input)

        msg = create_message(self.args, params)
        r_id = msg.signer_id
        for r in self.malicious_replicas:
            if not (r.id == r_id and r.check_certificate_validity(msg, self.malicious_ids)):
                self.attacker_reward += self.args.attacker_reward_invalid_certificate
                return None

        return msg

    def _replica_msg_to_malicious_input(self, msg):
        inputs = []
        # add msg_type
        num_msg_type = 11  # with noop and client
        inputs.extend(onehot(msg.msg_type, num_msg_type))
        # add signer_id
        inputs.extend(onehot(msg.signer_id, self.args.n_peers))
        # add view_num
        inputs.extend(onehot(msg.view_num, self.args.max_view_num))
        # add seq_num
        inputs.extend(onehot(msg.seq_num, self.args.max_seq_num))
        # add vals
        inputs.extend(onehot(msg.val, len(client_vals)))
        # add receiver_id
        inputs.extend(onehot(self._malicious_id_idx(msg.receiver_id), self.n_malicious))
        # add certificate
        if (msg.msg_type == type_dict["PrepareCertificate"] or
                msg.msg_type == type_dict["CommitCertificate"] or msg.msg_type == type_dict["NewView"]):
            inputs.extend(list_onehot(msg.certificate, self.args.n_peers))
        else:
            zeros = [0] * self.args.n_peers * 2
            inputs.extend(zeros)
        # print("msg_len in actual attacker obs: {}".format(len(inputs)))
        return inputs

    def _replica_msg_to_input(self, msg):
        inputs = []
        # add msg_type
        num_msg_type = 11  # with noop and client
        inputs.extend(onehot(msg.msg_type, num_msg_type))
        # add signer_id
        inputs.extend(onehot(msg.signer_id, self.args.n_peers))
        # add view_num
        inputs.extend(onehot(msg.view_num, self.args.max_view_num))
        # add seq_num
        inputs.extend(onehot(msg.seq_num, self.args.max_seq_num))
        # add vals
        inputs.extend(onehot(msg.val, len(client_vals)))
        # add receiver_id
        inputs.extend(onehot(msg.receiver_id, self.n_replicas))
        # add certificate
        if (msg.msg_type == type_dict["PrepareCertificate"] or
                msg.msg_type == type_dict["CommitCertificate"] or msg.msg_type == type_dict["NewView"]):
            inputs.extend(list_onehot(msg.certificate, self.args.n_peers))
        else:
            zeros = [0] * self.args.n_peers * 2
            inputs.extend(zeros)
        return inputs

    def _decoy_msgs(self, num, malicious=False):
        num_msg_type = 11
        msg_obs_space = int(num_msg_type + self.args.max_view_num +
                            self.args.max_seq_num + len(client_vals) + self.args.n_peers + self.args.n_peers * 2)
        msg_obs_space += self.n_malicious if malicious else self.args.n_peers
        return [0] * msg_obs_space * num

    def _malicious_id_idx(self, r_id):
        for idx in range(len(self.malicious_ids)):
            if self.malicious_ids[idx] == r_id:
                return idx
        print("Invalid id at _malicious_id_idx")
        return 0


"""
attacker action type: * self.args.max_message_num_per_round * self.num_malicious
    - message type: 0-9                                 num_action_space = 10  *9 is no-op
    - signer_id: self.malicious_ids                     num_action_space = self.args.num_malicious
    - view_num: 0-self.args.episode_limit/4             num_action_space = self.args.episode_limit/4  # max possible num_action_space
    - seq_num: 0-self.args.episode_limit/4              num_action_space = self.args.episode_limit/4
    - val: client_vals (0-3)                            num_action_space = len(client_vals) (4)                  
    - receiver_id: 0-self.args.n_peers-1                num_actions_space = self.args.n_peers
    - certificate: (0-self.args.n_peers-1)*2*self.f+1   num_actions_space = self.args.n_peers choose 0/1
    
total_action_space: self.args.max_message_num_per_round * self.num_malicious * 
            (10 + self.args.num_malicious + self.args.episode_limit/2 + 4 + self.args.n_peers + self.args.n_peers)

self.msg_type = type_dict[msg_type]
        self.view_num = view_num
        self.seq_num = seq_num
        self.signer_id = signer_id
        self.val = val
        self.receiver_id = receiver_id
        
attacker obs type: self.args.max_message_num_per_round * [self.args.episode_limit/2 + self.n_peers*2 + self.val + self.n_peers]
"""

"""
identifier action type：
    - self.args.n_peers
"""


def onehot(x, n):
    ret = [0] * n
    if x != float("inf") and 0 <= x < n:
        ret[x] = 1
    return ret


def rev_onehot(x):
    for idx in range(len(x)):
        if x[idx] == 1:
            return idx
    return -1


def list_onehot(x, n):  # for certificate
    ret = []
    for idx in range(n):
        if idx in x:  # chosen
            ret.extend([1, 0])
        else:
            ret.extend([0, 1])
    return ret


def rev_list_onehot(x):  # for certificates
    ret = []
    for idx in range(len(x) / 2):
        if x[2 * idx] == 1:  # chosen
            ret.append(idx)
    return ret
