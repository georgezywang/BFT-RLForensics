import copy

from protocols.PBFT.log import Log
from protocols.PBFT.message import create_message

state_dict = {"normal": 0, "view_changing": 1}
type_dict = {"PrePrepare": 1,
             "Prepare": 2,
             "Commit": 3,
             "ViewChange": 4,
             "NewView": 5,
             "PrepareCertificate": 6,
             "CommitCertificate": 7,
             "Client": 8,
             "RequestClient": 9}


class PBFTagent():
    def __init__(self, args, id):
        self.args = args
        self.id = id
        self.n_peers = args.n_peers
        self.current_view = 0
        self.changing_view = 0
        self.state = 0
        self.current_primary = 0
        self.primary_last_used_seq_num = 0
        # the latest stable seqnum known to this replica
        self.last_committed_seq_num = 0
        self.mainlog = None
        self.msgs_to_be_send = []
        self.idle_timer = float("inf")
        self.commit_timer = float("inf")
        self.view_change_timer = float("inf")

    def reset(self):
        self.current_primary = self.args.initialized_primary
        self.current_view = self.args.initialized_view_num
        self.state = state_dict["normal"]
        self.mainlog = Log(self.args)
        self.primary_last_used_seq_num = self.args.initialized_seq_num
        self.last_committed_seq_num = self.args.initialized_seq_num - 1
        self.msgs_to_be_send = []
        self.changing_view = float("inf")
        self.idle_timer = float("inf")
        self.commit_timer = float("inf")
        self.view_change_timer = float("inf")

    def handle_msgs(self, msgs):
        self.msgs_to_be_send = []

        # process all received messages at this round
        for msg in msgs:
            self.on_msg(msg)

        # update timers
        self.idle_timer -= 1
        self.commit_timer -= 1
        self.view_change_timer -= 1

        if self.idle_timer <= 0 or self.commit_timer <= 0 or self.view_change_timer <= 0:
            self._try_to_enter_view()

    def on_msg(self, msg):
        if msg.msg_type == type_dict["Client"]:
            self._on_client_msg(msg)
        if msg.msg_type == type_dict["PrePrepare"]:
            self._on_msg_preprepare(msg)
        if msg.msg_type == type_dict["Prepare"]:
            self._on_msg_prepare(msg)
        if msg.msg_type == type_dict["PrepareCertificate"]:
            self._on_msg_prepare_cert(msg)
        if msg.msg_type == type_dict["Commit"]:
            self._on_msg_commit(msg)
        if msg.msg_type == type_dict["CommitCertificate"]:
            self._on_msg_commit_cert(msg)
        if msg.msg_type == type_dict["ViewChange"]:
            self._on_msg_view_change(msg)
        if msg.msg_type == type_dict["NewView"]:
            self._on_msg_new_view(msg)

    def _on_client_msg(self, msg):
        if not self._is_normal_mode():
            return
        if self.last_committed_seq_num < msg.seq_num:
            if not self._is_current_primary():  # forward the received client message to current primary
                new_msg = copy.deepcopy(msg)
                new_msg.receiver_id = self.current_primary
                new_msg.signer_id = self.id
                self.msgs_to_be_send.append(new_msg)
            else:
                self._try_to_send_preprepare(msg)
        elif self.last_committed_seq_num == msg.seq_num:
            print("ignored!")  # FIXME: implement logging level
        else:
            print("ignored, i may be old!")

    def _try_to_send_preprepare(self, client_msg):
        if not self._is_current_primary() or not self._is_normal_mode():
            return
        if self.primary_last_used_seq_num + 1 > self.last_committed_seq_num + self.args.work_window_size:
            print("out of window!")
            return

        params = {"msg_type": "PrePrepare",
                  "view_num": self.current_view,
                  "seq_num": self.primary_last_used_seq_num + 1,
                  "signer_id": self.id,
                  "val": client_msg.val}
        self._create_broadcast_messages(params)

        params["receiver_id"] = self.id
        self_pp_msg = create_message(self.args, params)
        pp_msg_added = self.mainlog.get_entry(self_pp_msg.seq_num).add_message(self_pp_msg)

        self.primary_last_used_seq_num += 1

    def _on_msg_preprepare(self, msg):
        if msg.sender_id != self.current_primary:
            return

        if (self.mainlog.get_entry(msg.seq_num).is_preprepare_ready() and
                self.mainlog.get_entry(msg.seq_num).val != msg.val):
            self._try_to_enter_view()
            return

        if self._relevant_msg_for_active_view(msg):
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
                p_msg_added = self.mainlog.get_entry(msg.seq_num).add_message(create_message(self.args, params))
            # start idle timer
            self.idle_timer = self.args.idle_time_limit

    def _on_msg_prepare(self, msg):
        if msg.sender_id == self.current_primary:
            self._try_to_enter_view()
            return

        if self._relevant_msg_for_active_view(msg):
            if self.mainlog.get_entry(msg.seq_num).is_prepare_ready():
                # hey, the sender you are behind
                params = {"view_num": self.current_view,
                          "seq_num": msg.seq_num,
                          "signer_id": self.id,
                          "val": msg.val,
                          "receiver_id": msg.sender_id}
                if self.mainlog.get_entry(msg.seq_num).is_commit_ready():
                    params["msg_type"] = "CommitCertificate"
                else:
                    params["msg_type"] = "PrepareCertificate"
                self.msgs_to_be_send.append(create_message(self.args, params))
                return
            msg_added = self.mainlog.get_entry(msg.seq_num).add_message(msg)
            if msg_added:
                self._try_to_enter_commit(msg)

    def _try_to_enter_commit(self, msg):
        if not self.mainlog.get_entry(msg.seq_num).is_prepare_ready():
            return
        # send commit messages
        params = {"msg_type": "Commit",
                  "view_num": self.current_view,
                  "seq_num": msg.seq_num,
                  "signer_id": self.id,
                  "val": msg.val}
        self._create_broadcast_messages(params)
        params["receiver_id"] = self.id
        self.mainlog.get_entry(msg.seq_num).add_message(create_message(self.args, params))
        # start commit timer
        self.idle_timer = float("inf")
        self.commit_timer = self.args.commit_timer_limit

    def _on_msg_commit(self, msg):
        if self._relevant_msg_for_active_view(msg):
            if self.mainlog.get_entry(msg.seq_num).is_commit_ready():
                # hey, the sender you are behind
                params = {"msg_type": "CommitCertificate",
                          "view_num": self.current_view,
                          "seq_num": msg.seq_num,
                          "signer_id": self.id,
                          "val": msg.val,
                          "receiver_id": msg.sender_id}
                self.msgs_to_be_send.append(create_message(self.args, params))
                return
            msg_added = self.mainlog.get_entry(msg.seq_num).add_message(msg)
            if msg_added:
                self._try_to_commit(msg)

    def _try_to_commit(self, msg):
        if not self.mainlog.get_entry(msg.seq_num).is_commit_ready():
            return

        # send commit messages to client
        params = {"msg_type": "BlockCommit",
                  "view_num": self.current_view,
                  "seq_num": msg.seq_num,
                  "signer_id": self.id,
                  "val": msg.val,
                  "receiver_id": self.args.simulator_id}
        self.msgs_to_be_send.append(create_message(self.args, params))
        self.last_committed_seq_num = msg.seq_num
        self.commit_timer = float("inf")

    def _try_to_enter_view(self):
        self.state = state_dict["view_changing"]
        self.changing_view = self.current_view + 1
        self.idle_timer = float("inf")
        self.commit_timer = float("inf")
        self.view_change_timer = float("inf")
        params = {"msg_type": "ViewChange",
                  "view_num": self.changing_view,
                  "seq_num": float("inf"),
                  "signer_id": self.id,
                  "val": float("inf")}
        self._create_broadcast_messages(params)
        params["receiver_id"] = self.id
        self.mainlog.get_view_entry(self.changing_view).add_message(create_message(self.args, params))

    def _on_msg_view_change(self, msg):
        if self._is_valid_view_change(msg.view_num):
            if self.mainlog.get_view_entry(msg.view_num).is_view_change_ready():
                pass  # the sender is behind
            else:
                msg_added = self.mainlog.get_view_entry(msg.view_num).add_message(msg)
                if msg_added and self.mainlog.get_view_entry(msg.view_num).is_view_change_ready():
                    # ready to view change
                    self.state = state_dict["view_changing"]
                    self.changing_view = msg.view_num
                    if self._next_primary() != self.id:
                        self.view_change_timer = (msg.view_num - self.current_view) * self.args.view_change_duration
                    else:
                        params = {"msg_type": "NewView",
                                  "view_num": msg.view_num,
                                  "seq_num": self.last_committed_seq_num + 1,
                                  "signer_id": self.id,
                                  "val": msg.val,
                                  "certificate": self.mainlog.get_view_entry(self.changing_view).view_change_sigs}
                        self._create_broadcast_messages(params)
                        self._go_to_next_view(self.last_committed_seq_num + 1)

    def _on_msg_new_view(self, msg):
        if not self._is_normal_mode() and msg.sender_id == self._next_primary() and self._check_certificate(
                msg.certificate):
            self._go_to_next_view(msg.seq_num)

    def _go_to_next_view(self, new_seq_num):
        self.state = state_dict["normal"]
        self.current_view = self.changing_view
        self.changing_view = float("inf")
        self.current_primary = self._next_primary()
        self.view_change_timer = float("inf")
        self.mainlog.revert_log_to(new_seq_num)
        if self._is_current_primary():
            # request client to go back
            params = {"msg_type": "RequestClient",
                      "view_num": float("inf"),
                      "seq_num": self.last_committed_seq_num + 1,
                      "signer_id": self.id,
                      "val": float("inf")}
            self.msgs_to_be_send.append(create_message(self.args, params))

    def _on_msg_prepare_cert(self, msg):
        if self._relevant_msg_for_active_view(msg) and self._check_certificate(msg.certificate):
            msg_added = self.mainlog.get_entry.add_message(msg)
            if msg_added:
                self._try_to_enter_commit(msg)

    def _on_msg_commit_cert(self, msg):
        if self._relevant_msg_for_active_view(msg) and self._check_certificate(msg.certificate):
            msg_added = self.mainlog.get_entry.add_message(msg)
            if msg_added:
                self._try_to_commit(msg)

    def _is_valid_view_change(self, view_num):
        normal_change = self._is_normal_mode() and view_num > self.current_view
        view_change = not self._is_normal_mode() and view_num >= self.current_view + 1  # true when the node is a viewchange(current_view + 1)
        return normal_change or view_change

    def _is_normal_mode(self):
        return self.state == state_dict["normal"]  # FIXME: maintain a view manager

    def _relevant_msg_for_active_view(self, msg):
        if (self._is_normal_mode() and
                msg.view_num == self.current_view and
                self.last_committed_seq_num < msg.seq_num < self.last_committed_seq_num + self.args.work_window_size):
            return True
        return False

    def _create_broadcast_messages(self, params):
        for r_id in range(self.n_peers):
            if r_id != self.id:
                t_params = copy.deepcopy(params)
                t_params["receiver_id"] = r_id
                self.msgs_to_be_send.append(create_message(self.args, t_params))

    def _is_current_primary(self):
        return self.id == self.current_primary

    def _next_primary(self):
        return (self.current_primary + 1) % self.n_peers

    def _check_certificate(self, cert):
        return len(cert) >= 2 * self.args.f + 1
