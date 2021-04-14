import copy

from protocols.PBFT.log import Log
from protocols.PBFT.message import create_message


class PBFTagent():
    def __init__(self, args, id):
        self.args = args
        self.id = id
        self.n_peers = args.n_peers
        self.current_state = None
        self.current_view = 0
        self.current_view_active = False
        self.current_primary = 0
        self.current_seq_num = 0
        self.primary_last_used_seq_num = 0
        # the last seqnum used to generate a new seq (valid only when the replica is the primary)
        self.last_stable_seq_num = 0
        # the latest stable seqnum known to this replica
        self.seq_num_of_last_reply_to_client = -1
        self.mainlog = None
        self.msgs_to_be_send = []

    def initialize(self):
        self.current_primary = self.args.initialized_primary
        self.current_view = self.args.initialized_view_num
        self.current_view_active = True
        self.current_seq_num = self.args.innitialized_seq_num
        self.mainlog = Log(self.args)
        self.primary_last_used_seq_num = self.args.initialized_seq_num
        self.last_stable_seq_num = self.args.initialized_seq_num
        self.seq_num_of_last_reply_to_client = self.args.initialized_seq_num - 1
        self.msgs_to_be_send = []

    def handle_msgs(self, msgs):
        self.msgs_to_be_send = []
        for msg in msgs:
            self.on_msg(msg)

    def on_msg(self, msg):

    def on_client_msg(self, msg):
        if not self._current_view_is_active():
            return
        if self.seq_num_of_last_reply_to_client < msg.seq_num:
            if not self.is_current_primary():  # forward the received client message to current primary
                new_msg = copy.deepcopy(msg)
                new_msg.receiver_id = self.current_primary
                new_msg.signer_id = self.id
                self.msgs_to_be_send.append(new_msg)
            else:
                self._try_to_send_preprepare(msg)
        elif self.seq_num_of_last_reply_to_client == msg.seq_num:
            print("ignored!")  # FIXME: implement logging level
        else:
            print("ignored, i may be old!")

    def _try_to_send_preprepare(self, client_msg):

    def _on_msg_preprepare(self, msg):

    def _on_msg_prepare(self, msg):

    def _on_msg_commit(self, msg):

    def _try_to_enter_view(self):

    def _go_to_next_view(self):

    def _on_new_view(self):

    def _current_view_is_active(self):
        return self.current_view_active  # FIXME: maintain a view manager

    def _relevant_msg_for_active_window(self):

    def is_current_primary(self):
        return self.id == self.current_primary

