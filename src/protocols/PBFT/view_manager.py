state_dict = {"no_view": 0,
              "pending": 1,
              "pending_with_restrictions": 2,
              "in_view": 3}

class ViewManager():
    def __init__(self, args):
        self.state = None
        self.my_latest_active_view = 0
        self.my_latest_pending_view = 0
        self.collection_of_pp_msgs = {}
        self.view_change_msgs = {}
        self.new_view_msgs = {}
        self.view_change_msgs_of_pending_view = {}
        self.new_view_msgs_of_pending_view = {}
        self.min_seq_num_restriction_of_pending_view = 0
        self.max_seq_num_restriction_of_pending_view = 0
        self.lower_bound_seq_num_stable_for_pending_view = 0

    def reset(self):

    def is_view_acitve(self, view):
        return self.in_view() and self.my_latest_active_view == view

    def is_view_pending(self, view):
        return self.my_latest_pending_view == view and view > self.my_latest_active_view

    def waiting_for_msgs(self):
        return self.state == state_dict["pending_with_restrictions"]

    def add(self, msg):  # either new_view msg, view_change msg

    def compute_correct_relevant_view_num(self, out_max_known_correct_view, out_max_known_agreed_view):

    # should only be called when v >= my_latest_pending_view
    def has_new_view_messge(self, view):

    def in_view(self):
        return self.state == state_dict["in_view"]

