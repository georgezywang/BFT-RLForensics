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
    elif msg_type == "PrepareCertificate" or msg_type == "CommitCertificate" or msg_type == "NewView":
        return CertificateMsg(args=args,
                              msg_type=msg_type,
                              view_num=params["view_num"],
                              seq_num=params["seq_num"],
                              signer_id=params["signer_id"],
                              val=params["val"],
                              receiver_id=params["receiver_id"],
                              certificate=params["certificate"])


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
    def __init__(self, args, view_num, seq_num, signer_id, val, receiver_id, simulator_id):
        super().__init__(args, "Client", view_num, seq_num, signer_id, val, receiver_id)
        self.simulator_id = simulator_id


class CertificateMsg(PBFTMessage):
    def __init__(self, args, view_num, seq_num, signer_id, val, receiver_id, certificate, msg_type):
        super().__init__(args, msg_type, view_num, seq_num, signer_id, val, receiver_id)
        self.cert = certificate


class ViewChangeMsg(PBFTMessage):
    def __init__(self, args, view_num, signer_id, receiver_id):
        super().__init__(args, "ViewChange", view_num, float("inf"), signer_id, float("inf"), receiver_id)


class RequestClient(PBFTMessage):
    def __init__(self, args, view_num, seq_num, signer_id, val):
        super().__init__(args, "RequestClient", view_num, seq_num, signer_id, val, args.simulator_id)
        # Please sync the request back to seq_num
