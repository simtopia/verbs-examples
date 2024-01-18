import verbs
from verbs.sim import BaseAgent


class AdminAgent(BaseAgent):
    def __init__(
        self,
        network,
        i: int,
    ):

        address = verbs.utils.int_to_address(i)
        self.deploy(network, address, 0)

    def update(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass
