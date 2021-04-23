import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, output_shape)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        # h = F.relu(self.fc3(h))
        q = self.fc2(h)
        return q, h


class RNNIdentifierAgent(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(RNNIdentifierAgent, self).__init__()
        self.args = args
        self.rnn = RNNAgent(input_shape, output_shape, args)

    def init_hidden(self):
        return self.rnn.init_hidden()

    def forward(self, inputs, hidden_state):
        x, h = self.rnn(inputs, hidden_state)
        x = F.normalize(x, dim=-1)  # normalize to range [0, 1]
        return x, h


class RNNAttackerAgent(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(RNNAttackerAgent, self).__init__()
        self.args = args
        self.rnn = RNNAgent(input_shape, output_shape, args)
        self.msg_action_shape = self._get_msg_shape()

    def init_hidden(self):
        return self.rnn.init_hidden()

    def forward(self, inputs, hidden_state):
        print("inputs shape: {}".format(inputs.shape))
        x, h = self.rnn(inputs, hidden_state)
        # time to dissemble x
        num_msg_type = 10
        x = x.view(-1, self.args.max_message_num_per_round*self.args.num_malicious, self.msg_action_shape)  # split
        msg_types, signer_ids, view_nums, seq_nums, vals, receiver_ids, certificates = x.split([num_msg_type,
                                                                                                self.args.num_malicious,
                                                                                                self.args.max_view_num,
                                                                                                self.args.max_seq_num,
                                                                                                self.args.total_client_vals,
                                                                                                self.args.n_peers,
                                                                                                self.args.n_peers*2], dim=2)
        certificates = list(certificates.split(2, dim=-1))  # make tuple iterable ([bs, max_msg_num, 2])*n_peers
        msg_types = F.softmax(msg_types, dim=-1)
        signer_ids = F.softmax(signer_ids, dim=-1)
        view_nums = F.softmax(view_nums, dim=-1)
        seq_nums = F.softmax(seq_nums, dim=-1)
        vals = F.softmax(vals, dim=-1)
        # print("vals shape: {}, vals: {}".format(vals.shape, vals))
        receiver_ids = F.softmax(receiver_ids, dim=-1)
        for idx in range(len(certificates)):
            # print("sig shape: {}".format(certificates[idx].shape))
            # print("sig in certificate: {}".format(certificates[idx]))
            certificates[idx] = F.softmax(certificates[idx], dim=-1)
        x = (msg_types, signer_ids, view_nums, seq_nums, vals, receiver_ids, certificates)
        return x, h

    def _get_msg_shape(self):  # TODO: move this to env_info[]
        num_msg_type = 10  # no client type, 9 is no-op
        msg_action_space = num_msg_type + self.args.num_malicious + \
                           self.args.max_seq_num + self.args.max_view_num + \
                           self.args.total_client_vals + self.args.n_peers + self.args.n_peers*2
        return int(msg_action_space)
