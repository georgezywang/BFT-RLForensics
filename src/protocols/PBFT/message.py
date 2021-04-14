type_dict = {"PrePrepare": 1,
             "Prepare": 2,
             "Commit": 3,
             "ViewChange": 4,
             "NewView": 5,
             "Seal": 6,
             "SealRequest": 7,
             "Client": 8}

def create_message(args, params):
    msg_type = params["msg_type"]
    if msg_type == "Client":
        return ClientMsg(args=args,
                         view_num=params["view_num"],
                         seq_num=params["seq_num"],
                         signer_id=params["signer_id"],
                         val=params["val"],
                         receiver_id=params["receiver_id"],
                         client_id=params["client_id"])

class PBFTMessage():
    def __init__(self, args, msg_type, view_num, seq_num, signer_id, val, receiver_id):
        self.args = args
        self.msg_type = type_dict[msg_type]
        self.view_num = view_num
        self.seq_num = seq_num
        self.signer_id = signer_id
        self.val = val
        self.receiver_id = receiver_id

class ClientMsg(PBFTMessage):
    def __init__(self, args, view_num, seq_num, signer_id, val, receiver_id, client_id):
        super().__init__(args, "Client", view_num, seq_num, signer_id, val, receiver_id)
        self.client_id = client_id



