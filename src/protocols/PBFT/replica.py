import copy

from protocols.PBFT.log import Log
from protocols.PBFT.message import create_message
from protocols.PBFT.view_manager import ViewManager


class PBFTagent():
    def __init__(self, args, id):
        self.args = args
        self.id = id
        self.n_peers = args.n_peers
        self.current_view = 0
        self.current_view_active = False
        self.current_primary = 0
        self.primary_last_used_seq_num = 0
        # the last seqnum used to generate a new seq (valid only when the replica is the primary)
        self.last_stable_seq_num = 0
        # the latest stable seqnum known to this replica
        self.last_executed_seq_num = 0
        self.seq_num_of_last_reply_to_client = -1
        self.mainlog = None
        self.view_manager = None
        self.msgs_to_be_send = []

    def reset(self):
        self.current_primary = self.args.initialized_primary
        self.current_view = self.args.initialized_view_num
        self.current_view_active = True
        self.mainlog = Log(self.args)
        self.view_manager = ViewManager(self.args)
        self.primary_last_used_seq_num = self.args.initialized_seq_num
        self.last_stable_seq_num = self.args.initialized_seq_num - 1
        self.last_executed_seq_num = self.args.initialized_seq_num - 1
        self.seq_num_of_last_reply_to_client = self.args.initialized_seq_num - 1
        self.msgs_to_be_send = []

    def handle_msgs(self, msgs):
        self.msgs_to_be_send = []
        for msg in msgs:
            self.on_msg(msg)

    def on_msg(self, msg):

    def _on_client_msg(self, msg):
        if not self._is_current_view_active():
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
        if not self.is_current_primary() or not self._is_current_view_active():
            return
        if self.primary_last_used_seq_num + 1 > self.last_stable_seq_num + self.args.work_window_size:
            print("out of window!")
            return
        if self.primary_last_used_seq_num > self.last_executed_seq_num:
            print("huh, last used seq num > last executed")
            return

        params = {"msg_type": "PrePrepare",
                  "view_num": self.current_view,
                  "seq_num": self.primary_last_used_seq_num+1,
                  "signer_id": self.id,
                  "val": client_msg.val}
        self._create_broadcast_messages(params)

        params["receiver_id"] = self.id
        self_pp_msg = create_message(self.args, params)
        pp_msg_added = self.mainlog.get_entry(self_pp_msg.seq_num).add_message(self_pp_msg)
        self.primary_last_used_seq_num += 1

    def _on_msg_preprepare(self, msg):
        if (not self._is_current_view_active() and
            self.view_manager.waiting_for_msgs() and
            msg.seq_num > self.last_stable_seq_num):
            if self.view_manager.add_potentially_missing_pp(msg, last_stable_seq_num):
                self._try_to_enter_view()
            else:
                print("pp discarded!")
            return

        if self._relevant_msg_for_active_view(msg) and msg.sender_id == self.current_primary:
            pp_msg_added = self.mainlog.get_entry(msg.seq_num).add_message(msg)
            if pp_msg_added:
                # send prepare
                params = {"msg_type": "Prepare",
                          "view_num": self.current_view,
                          "seq_num": msg.seq_num,
                          "signer_id": self.id,
                          "val": msg.val}
                self._create_broadcast_messages(params)
                params["receiver_id"] = self.id
                p_msg_added = self.mainlog.get_entry(msg.seq_num).add_message(create_message(params))
                
    def _on_msg_prepare(self, msg):


    def _on_msg_commit(self, msg):


    def _try_to_enter_view(self):


    def _go_to_next_view(self):


    def _on_new_view(self):


    def _is_current_view_active(self):
        return self.view_manager.is_view_active(self.current_view)  # FIXME: maintain a view manager

    def _relevant_msg_for_active_view(self, msg):
        msg_seq_num = msg.seq_num
        msg_view_num = msg.view_num

        if (self._is_current_view_active() and
                msg_view_num == self.current_view and
                self.last_stable_seq_num < msg_seq_num < self.last_stable_seq_num + self.args.work_window_size):
            return True
        return False

    def _create_broadcast_messages(self, params):
        for r_id in range(self.n_peers):
            if r_id != self.id:
                t_params = copy.deepcopy(params)
                t_params["receiver_id"] = r_id
                self.msgs_to_be_send.append(create_message(self.args, t_params))

    def is_current_primary(self):
        return self.id == self.current_primary

