import verbs
from verbs.sim import BaseAgent


class AdminAgent(BaseAgent):
    def __init__(
        self,
        env,
        i: int,
    ):

        address = verbs.utils.int_to_address(i)
        self.deploy(env, address, 0)

    def update(self, *args):
        pass

    def record(self, *args):
        pass
