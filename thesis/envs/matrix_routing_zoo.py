from alpyne.data.spaces import Observation
from .randdispatcher import StationDispatcher
from .matrix_zoo import MatrixMA

counter = [0]
global_client = [
    None,
]


class MatrixRoutingMA(MatrixMA):
    """
    Pettingzoo Env for each AGV routing when it needs to.

    model_path: If supposed to build its own client
    client: If client is given
    max_fleetsize: if None, no shuffling is done, observations are taken as they are.
        If max_fleetsize is given (should be >= fleetsize) the obs is always
        max_fleetsize agvs big and the indices are randomly allocated (shuffled every 100 steps)
    dispatcher: If None, RandDispatcher is used
    counter: reference to a List<int>, first entry is used as port/seed and the incremented
    """

    def __init__(
        self,
        model_path: str = None,
        startport=51150,
        fleetsize: int = 1,
        max_fleetsize: int = 1,
        config_args: dict = dict(),
        dispatcher=None,
        max_steps: int = None,
        max_seconds: int = None,
        verbose=False,
    ):
        if dispatcher is None:
            self.dispatcher = StationDispatcher()
        else:
            self.dispatcher = dispatcher
        config_args["dispatchingOnArrival"] = True
        super().__init__(
            model_path,
            startport,
            fleetsize,
            max_fleetsize,
            config_args,
            max_steps,
            max_seconds,
            verbose,
        )

    def _catch_nontraining(self, observation: Observation) -> Observation:
        while observation.caller.endswith("Dispatching"):
            action = self.dispatcher(observation)
            self.sim.take_action(action)
            self.sim.wait_for_completion()
            observation = self.sim.get_observation()
        return observation

    def get_config(self, config_args):
        conf = super().get_config(config_args)
        conf.runmode = 4
        return conf