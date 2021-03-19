from controller.shared_controller import BasicMAC
from controller.shared_controller_nash_q import SharedNashMAC

REGISTRY = {}

REGISTRY["shared_nash"] = SharedNashMAC
REGISTRY["shared"] = BasicMAC