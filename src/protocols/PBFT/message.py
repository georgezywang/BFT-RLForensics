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


def create_message(args, params):
    if isinstance(params["msg_type"], int):
        inv_dict = {v: k for k, v in type_dict.items()}
        msg_type = inv_dict[params["msg_type"]]
    else:
        msg_type = params["msg_type"]
    if msg_type == "Client":
        return ClientMsg(args=args,
                         view_num=params["view_num"],
                         seq_num=params["seq_num"],
                         signer_id=params["signer_id"],
                         val=params["val"],
                         receiver_id=params["receiver_id"])
    elif msg_type == "PrepareCertificate" or msg_type == "CommitCertificate" or msg_type == "NewView":
        return CertificateMsg(args=args,
                              msg_type=msg_type,
                              view_num=params["view_num"],
                              seq_num=params["seq_num"],
                              signer_id=params["signer_id"],
                              val=params["val"],
                              receiver_id=params["receiver_id"],
                              certificate=params["certificate"])
    elif msg_type == "RequestClient":
        return RequestClientMsg(args=args,
                                view_num=params["view_num"],
                                seq_num=params["seq_num"],
                                signer_id=params["signer_id"],
                                val=params["val"])
    elif msg_type == "ViewChange":
        return ViewChangeMsg(args=args,
                             view_num=params["view_num"],
                             signer_id=params["signer_id"],
                             receiver_id=params["receiver_id"])

    else:
        return PBFTMessage(args=args,
                           msg_type=msg_type,
                           view_num=params["view_num"],
                           seq_num=params["seq_num"],
                           signer_id=params["signer_id"],
                           val=params["val"],
                           receiver_id=params["receiver_id"])

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
    def __init__(self, args, view_num, seq_num, signer_id, val, receiver_id):
        super().__init__(args, "Client", view_num, seq_num, signer_id, val, receiver_id)


class CertificateMsg(PBFTMessage):
    def __init__(self, args, view_num, seq_num, signer_id, val, receiver_id, certificate, msg_type):
        super().__init__(args, msg_type, view_num, seq_num, signer_id, val, receiver_id)
        self.cert = certificate


class ViewChangeMsg(PBFTMessage):
    def __init__(self, args, view_num, signer_id, receiver_id):
        super().__init__(args, "ViewChange", view_num, float("inf"), signer_id, float("inf"), receiver_id)


class RequestClientMsg(PBFTMessage):
    def __init__(self, args, view_num, seq_num, signer_id, val):
        super().__init__(args, "RequestClient", view_num, seq_num, signer_id, val, args.simulator_id)
        # Please sync the request back to seq_num
