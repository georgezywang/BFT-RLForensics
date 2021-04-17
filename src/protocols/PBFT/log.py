import copy


class Log():
    def __init__(self, args):
        self.args = args
        self.entries = {}
        self.view_change_entries = {}

    def get_entry(self, seq_num):
        if seq_num not in self.entries.keys():
            self.entries[seq_num] = Entry(self.args, seq_num)
        return self.entries[seq_num]

    def get_view_entry(self, view_num):
        if view_num not in self.entries.keys():
            self.view_change_entries[view_num] = ViewEntry(self.args, view_num)
        return self.view_change_entries[view_num]

    def revert_log_to(self, new_seq_num):
        revert = [seq_num for seq_num in self.entries.keys() if seq_num >= new_seq_num]
        for seq_num in revert:
            del self.entries[seq_num]


type_dict = {"PrePrepare": 1,
             "Prepare": 2,
             "Commit": 3,
             "ViewChange": 4,
             "NewView": 5,
             "PrepareCertificate": 6,
             "CommitCertificate": 7,
             "Client": 8,
             "RequestClient": 9}


class Entry():
    def __init__(self, args, seq_num):
        self.args = args
        self.seq_num = seq_num
        self.val = float("inf")
        self.prepare_sigs = []
        self.commit_sigs = []

    def add_message(self, msg):
        if msg.seq_num != self.seq_num:
            print("Sequence number of message and log not matching, ignored (Message {}, Log entry {})".format(
                msg.seq_num, self.seq_num))
            return
        if msg.type == type_dict["PrePrepare"]:
            return self._add_preprepare(msg)
        if msg.type == type_dict["Prepare"]:
            return self._add_prepare(msg)
        if msg.type == type_dict["Commit"]:
            return self._add_commit(msg)
        if msg.type == type_dict["PrepareCertificate"]:
            return self._add_prepare_cert(msg)
        if msg.type == type_dict["CommitCertificate"]:
            return self._add_commit_cert(msg)

    def is_preprepare_ready(self):
        return self.val != float("inf")

    def is_prepare_ready(self):
        return len(self.prepare_sigs) >= 2 * self.args.f + 1

    def is_commit_ready(self):
        return len(self.commit_sigs) >= 2 * self.args.f + 1

    def _add_preprepare(self, msg):
        if self.is_preprepare_ready():
            return False
        self.val = msg.val
        return True

    def _add_prepare(self, msg):
        if msg.val != self.val:
            return False
        if msg.signer_id in self.prepare_sigs:
            return False
        self.prepare_sigs.append(msg.signer_id)
        return True

    def _add_commit(self, msg):
        if msg.val != self.val:
            return False
        if msg.signer_id in self.commit_sigs:
            return False
        self.commit_sigs.append(msg.signer_id)
        return True

    def _add_prepare_cert(self, msg):
        self.prepare_sigs = copy.deepcopy(msg.certificate)
        self.val = msg.val
        return True

    def _add_commit_cert(self, msg):
        self.commit_sigs = copy.deepcopy(msg.certificate)
        self.val = msg.val
        return True


class ViewEntry():
    def __init__(self, args, view_num):
        self.args = args
        self.view_num = view_num
        self.view_change_sigs = []

    def add_message(self, msg):
        if msg.view_num != self.view_num:
            print("Sequence number of message and log not matching, ignored (Message {}, Log entry {})".format(
                msg.view_num, self.view_num))
            return
        if msg.type != type_dict["ViewChange"]:
            print("Message type in ViewChange, NewView, Seal, or SealRequest, ignored")
            return
        self._add_view_change(msg)

    def is_view_change_ready(self):
        return len(self.view_change_sigs) >= 2 * self.args.f + 1

    def _add_view_change(self, msg):
        if msg.signer_id in self.view_change_sigs:
            return False
        self.view_change_sigs.append(msg.signer_id)
        return True


class ClientLog():
    def __init__(self, args):
        self.args = args
        self.entries = {}

    def get_entry(self, seq_num):
        if seq_num not in self.entries.keys():
            self.entries[seq_num] = ConsensusEntry(self.args, seq_num)
        return self.entries[seq_num]


class ConsensusEntry():
    def __init__(self, args, seq_num):
        self.args = args
        self.seq_num = seq_num
        self.val_sigs = {}
        self.visited = False

    def add_message(self, msg):
        if msg.seq_num != self.seq_num:
            return
        if msg.type != type_dict["BlockCommit"]:
            print("Message type in ViewChange, NewView, Seal, or SealRequest, ignored")
            return
        self._add_block_commit(msg)

    def get_block_committed_val(self):
        ret = []
        for val in self.val_sigs.keys():
            if len(self.val_sigs[val]) >= 2 * self.args.f + 1:
                ret.append(val)
        if len(ret) != 0 and self.visited == False:
            self.visited = True
        return ret

    def _add_block_commit(self, msg):
        if msg.val in self.val_sigs.keys():
            if msg.signer_id not in self.val_sigs[msg.val]:
                self.val_sigs[msg.val].append(msg.signer_id)
                return True
        else:
            self.val_sigs[msg.val] = [msg.signer_id]
            return True
        return False
