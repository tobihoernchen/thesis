import statistics
from typing import OrderedDict
from alpyne.data.spaces import Observation, Action
from .matrix_zoo import MatrixMA
from .randdispatcher import StationDispatcher
from gym import spaces
import torch
import numpy as np


counter = [0]
global_client = [
    None,
]


class MatrixDispatchingMA(MatrixMA):
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
        self.general_reward = 0.1
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

    def get_config(self, config_args):
        conf = super().get_config(config_args)
        conf.runmode = 4
        conf.dispatchingOnArrival = True
        #conf.obsdisp_includeAgvTarget = True
        return conf

    def get_agents(self):
        agents_routing = super().get_agents()
        agents_dispatching = [agent + "_Dispatching" for agent in agents_routing]
        return agents_routing + agents_dispatching

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

    def _save_observation(self, observation: Observation):
        agent = observation.caller
        self.agent_selection = agent
        self.rewards[self.agent_selection] = observation.rew[0] + self.general_reward

        # team rewards
        # for agent in self.agents:
        #    self.rewards[agent] += observation.rew[0] / 50

        self._cumulative_rewards[self.agent_selection] += observation.rew[0]
        obs_agent = torch.Tensor(observation.obs)
        self._save_for_agent(obs_agent, agent, int(agent.split("_")[0]))

    def _save_for_agent(self, obs_complete: torch.Tensor, agent: str, i: int):
        awaiting_orders = obs_complete[i, 1]
        inSystem = obs_complete[i, 0]

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
        action = action if action is not None else 0
        if self.val_nodes is not None and agent.endswith("Dispatching"):
            action = list(self.val_nodes.keys())[action]
        iagent = (
            int(agent.split("_")[0]) + 1000
            if agent.endswith("Dispatching")
            else int(agent)
        )
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
                    value=iagent,
                    unit=None,
                ),
            ]
        )
