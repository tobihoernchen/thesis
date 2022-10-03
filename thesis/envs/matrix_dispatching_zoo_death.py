import statistics
from typing import OrderedDict
from alpyne.data.spaces import Observation, Action
from .matrix_zoo import MatrixMA
from .randdispatcher import StationDispatcher
from gym import spaces
import torch
import numpy as np
from alpyne.client.abstract import BaseAlpyneEnv
from alpyne.client.alpyne_client import AlpyneClient
from alpyne.data.spaces import Configuration, Observation, Action
from gym import spaces
from typing import OrderedDict, Union, Tuple, Dict, Optional, NoReturn
import torch
import random
import numpy as np
from .randdispatcher import RandDispatcher
from ..utils.build_config import build_config
from .base_alpyne_zoo import BaseAlpyneZoo

counter = [0]
global_client = [
    None,
]


class MatrixDispatchingMA(BaseAlpyneZoo):
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
        n_stations=5,
        max_fleetsize: int = 1,
        config_args: dict = dict(),
        dispatcher=None,
        max_steps: int = None,
        max_seconds: int = None,
        verbose=False,
    ):
        self.fleetsize = fleetsize
        self.max_fleetsize = max_fleetsize

        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.verbose = verbose

        self.stepcounter = 0
        self.context = None

        self.possible_agents = self.get_agents()
        self.agents = [str(i) for i in range(self.fleetsize)] + [
            str(i) + "_Dispatching" for i in range(self.fleetsize)
        ]
        self.current_alias = list(range(self.fleetsize))

        self.metadata = dict(is_parallelizable=True)
        self.statistics = None

        self.shuffle()

        self.client = None
        self.started = False
        self.model_path = model_path
        self.startport = startport
        if "worker_index" in config_args.keys():
            self.startport += 4 * config_args["worker_index"]

        self.config = self.get_config(config_args)

        config_args["dispatchingOnArrival"] = True
        self.shape_agvs = None
        self.shape_stat = None
        self.n_nodes = None
        self.n_stations = None
        self.val_nodes = None
        if dispatcher == "stat":
            self.dispatcher = StationDispatcher()
        else:
            self.dispatcher = dispatcher
        self.agent_hwm = self.fleetsize
        self.agent_dict = {}

        super().__init__(None, self.agents)

    def get_config(self, config_args):
        conf = build_config(config_args, self.fleetsize)
        conf.reward_separateAgv = True
        conf.routingOnNode = True
        conf.obs_includeNodesInReach = True
        conf.runmode = 4
        conf.dispatchingOnArrival = True
        # conf.obsdisp_includeAgvTarget = True
        return conf

    def get_agents(self):
        agents_routing = [str(i) for i in range(1000)]
        agents_dispatching = [agent + "_Dispatching" for agent in agents_routing]
        return agents_routing + agents_dispatching

    def start(self):
        if global_client[0] is None:
            port = self.startport + int(counter[0])
            counter[0] = counter[0] + 1  # increment to change port for all envs created
            global_client[0] = AlpyneClient(self.model_path, port=port, verbose=False)

        self.client = global_client[0]
        self.run = self.client.create_reinforcement_learning(self.config)
        self.started = True
        super().start(self.run)

    def shuffle(self):
        self.shufflerule = random.sample(
            list(range(self.max_fleetsize - 1)), self.fleetsize - 1
        )

    def seed(self, seed: int = None):
        if self.verbose:
            print(f"seeded with {seed}")
        if seed is not None:
            self.config.seed = seed
        else:
            self.config.seed = random.randint(0, 1000)

    def reset(
        self,
        config: Configuration = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> "BaseAlpyneZoo.PyObservationType":
        self.agents = [str(i) for i in range(self.fleetsize)] + [
            str(i) + "_Dispatching" for i in range(self.fleetsize)
        ]
        self.current_alias = list(range(self.fleetsize))

        if self.verbose:
            print(f"reset")
        if not self.started:
            self.start()
        self.shuffle()
        self.stepcounter = 0
        self.seed(seed=seed)
        return super().reset(self.config, seed, return_info, options)

    def step(
        self, action: "BaseAlpyneZoo.PyActionType"
    ) -> Tuple["BaseAlpyneZoo.PyObservationType", float, bool, Optional[dict]]:
        if self.verbose:
            print(f"step with {action}")
        self.stepcounter += 1

        if self.stepcounter % 100 == 0:
            self.shuffle()

        return super().step(action)

    def last(self, observe=True) -> "BaseAlpyneZoo.PyObservationType":
        if self.verbose:
            print(f"last called. done {self.dones}")
        return super().last(observe)

    def _get_observation_space(self, agent) -> spaces.Space:
        """Describe the dimensions and bounds of the observation"""
        if self.shape_agvs is None:
            self.reset()
        return spaces.Dict(
            {
                "agvs": spaces.Box(
                    low=0, high=1, shape=(self.shape_agvs[0] * self.shape_agvs[1],)
                ),
                "stations": spaces.Box(
                    low=0, high=1, shape=(self.shape_stat[0] * self.shape_stat[1],)
                ),
                "action_mask": spaces.Box(low=0, high=1, shape=(self.n_nodes,)),
            }
        )

    def _get_action_space(self, agent) -> spaces.Space:
        return spaces.Discrete(self.n_nodes)

    def last(self, observe=True) -> "BaseAlpyneZoo.PyObservationType":
        value = super().last(observe)
        self.rewards[self.agent_selection] = 0
        return value

    def _save_observation(self, observation: Observation):
        if observation.caller.endswith("Dispatching"):
            agent_i = self.current_alias[int(observation.caller.split("_")[0])]
            agent = str(agent_i) + "_Dispatching"
        else:
            agent_i = self.current_alias[int(observation.caller)]
            agent = str(agent_i)
        self.agent_selection = agent
        self.rewards[self.agent_selection] = observation.rew[0]

        # team rewards
        # for agent in self.agents:
        #    self.rewards[agent] += observation.rew[0] / 50

        obs_agent = torch.Tensor(observation.obs)
        self._save_for_agent(obs_agent, agent, self.current_alias.index(agent_i))

    def _save_for_agent(self, obs_complete: torch.Tensor, agent: str, i: int):
        inSystem = obs_complete[i, 0]
        if not inSystem and not self.dones[str(self.current_alias[i])]:
            self.dones[str(self.current_alias[i])] = True
            self.dones[str(self.current_alias[i]) + "_Dispatching"] = True
        if (
            inSystem
            and self.dones[str(self.current_alias[i])]
            and not all(self.dones.values())
        ):
            self.remove_agent(str(self.current_alias[i]))
            self.remove_agent(str(self.current_alias[i]) + "_Dispatching")
            self.add_agent(str(self.agent_hwm))
            self.add_agent(str(self.agent_hwm) + "_Dispatching")
            self.current_alias[i] = self.agent_hwm
            agent = (
                str(self.agent_hwm)
                if not agent.endswith("Dispatching")
                else str(self.agent_hwm) + "_Dispatching"
            )
            self.agent_selection = agent
            self.agent_hwm += 1
        nrows = self.shape_agvs[0] + self.shape_stat[0]
        ncols = self.shape_agvs[1]
        obs_converted = torch.zeros((nrows, ncols))
        if inSystem:
            obs_converted[0, 0] = 1  # Attention encoding for the AGV to decide for
            obs_converted[0, 1 : obs_complete.shape[1] + 1] = obs_complete[i]
            obs_complete = obs_complete[[x != i for x in range(len(obs_complete))]]
            for src, tgt in enumerate(self.shufflerule):
                obs_converted[tgt + 1, 1 : obs_complete.shape[1] + 1] = obs_complete[
                    src
                ]
            obs_converted[1:, :2] = 0
            obs_converted[1:, 6:] = 0
            if obs_complete.shape[0] > self.fleetsize - 1:
                obs_converted[
                    self.max_fleetsize :,
                    1 : obs_complete.shape[1] + 1,
                ] = obs_complete[self.fleetsize - 1 :]
        if agent.endswith("Dispatching"):
            mask = [
                1.0,
            ] * self.n_nodes
        else:
            mask = (
                [
                    1.0,
                ]
                + [
                    1.0 if not torch.all(act == 0) else 0.0
                    for act in obs_converted[0, 8:16].reshape(4, 2)
                ]
                + [
                    0.0,
                ]
                * (self.n_nodes - 5)
            )
        self.observations[agent] = dict(
            agvs=np.array(obs_converted[: self.max_fleetsize].flatten()),
            stations=np.array(obs_converted[self.max_fleetsize :].flatten()),
            action_mask=np.array(mask, dtype=np.float32),
        )

    def _convert_to_action(self, action, agent):
        """Convert the action sent as part of the Gym interface to an Alpyne Action object"""
        agent = self.agent_selection
        action = action if action is not None else 0
        if self.val_nodes is not None and agent.endswith("Dispatching"):
            action = list(self.val_nodes.keys())[action]
        iagent = self.current_alias.index(int(agent.split("_")[0]))
        outagent = iagent + 1000 if agent.endswith("Dispatching") else iagent

        return Action(
            data=[
                dict(
                    name="actions",
                    type="INTEGER_ARRAY",
                    value=[
                        action,
                    ],
                    unit=None,
                ),
                dict(
                    name="receiver",
                    type="INTEGER",
                    value=outagent,
                    unit=None,
                ),
            ]
        )

    def _terminal_alternative(self, observation: Observation) -> bool:
        """Optional method to add *extra* terminating conditions"""
        terminal_max_steps = (
            (self.stepcounter >= self.max_steps)
            if self.max_steps is not None
            else False
        )
        time = 0 if self.max_seconds is None else self.run.get_state()[1]["model_time"]
        terminal_max_seconds = (
            time >= self.max_seconds if self.max_seconds is not None else False
        )
        return terminal_max_steps or terminal_max_seconds

    def close(self):
        self.sim.stop()

    def _catch_nontraining(self, observation: Observation) -> Observation:

        if "statTitles" in observation.names():
            self.statistics = {
                title: value
                for title, value in zip(observation.statTitles, observation.statValues)
            }
            if "rout_len" in self.statistics.keys():
                features = (
                    int(max(self.statistics["rout_len"], self.statistics["disp_len"]))
                    + 1
                )
                self.shape_agvs = (self.max_fleetsize, features)
                self.shape_stat = (int(self.statistics["n_stat"]), features)
        if self.n_nodes is None and "n_nodes" in observation.names():
            self.n_nodes = int(observation.n_nodes)

        if (
            "networkcontext" in observation.names()
            and observation.networkcontext is not None
        ):
            nodes = [node for node in observation.networkcontext if node[3] == 0]
            self.val_nodes = OrderedDict(
                [[i, node[:2]] for i, node in enumerate(nodes) if node[2] == 1]
            )
            self.n_nodes = len(self.val_nodes)

        if self.dispatcher is not None:
            while observation.caller.endswith("Dispatching"):
                action = self.dispatcher(observation)
                self.sim.take_action(action)
                self.sim.wait_for_completion()
                observation = self.sim.get_observation()
        return observation
