class Log():
    def __init__(self, args):
        self.args = args
        self.entries = {}

    def get_entry(self, seq_num):
        if seq_num in self.entries.keys():
            return self.entries[seq_num]
        else:
            self.entries[seq_num] = Entry(self.args, seq_num)

type_dict = {"PrePrepare": 1,
             "Prepare": 2,
             "Commit": 3,
             "ViewChange": 4,
             "NewView": 5,
             "Seal": 6,
             "SealRequest": 7,
             "Client": 8}

class Entry():
    def __init__(self, args, seq_num):
        self.args = args
        self.seq_num = seq_num
        self.val = float("inf")
        self.prepare_sigs = []
        self.commit_sigs = []

    def add_message(self, msg):
        if msg.seq_num != self.seq_num:
            print("Sequence number of message and log not matching, ignored (Message {}, Log entry {})".format(msg.seq_num, self.seq_num))
            return
        if msg.type > 3:
            print("Message type in ViewChange, NewView, Seal, or SealRequest, ignored")
            return
        if msg.type == 1:
            self._add_preprepare(msg)
        if msg.type == 2:
            self._add_prepare(msg)
        if msg.type == 3:
            self._add_commit(msg)

    def is_preprepare_ready(self):
        return self.val != float("inf")

    def is_prepare_ready(self):
        return len(self.prepare_sigs) >= 2*self.args.f+1

    def is_commit_ready(self):
        return len(self.commit_sigs) >= 2*self.args.f + 1

    def _add_preprepare(self, msg):
        self.val = msg.val

    def _add_prepare(self, msg):
        if msg.val != self.val:
            print("Current primary of Prepare message and log not matching (Message {}, Log entry {})".format(
                msg.val,
                self.val))
            return
        self.prepare_sigs.append(msg.signer_id)

    def _add_commit(self, msg):
        if msg.val != self.val:
            print("Current primary of Commit message and log not matching (Message {}, Log entry {})".format(
                msg.val,
                self.val))
            return
        self.commit_sigs.append(msg.signer_id)


