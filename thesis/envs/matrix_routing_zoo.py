from alpyne.client.abstract import BaseAlpyneEnv
from alpyne.client.alpyne_client import AlpyneClient
from alpyne.data.spaces import Configuration, Observation, Action
from gym import spaces
from typing import Union, Tuple, Dict, Optional, NoReturn
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


class MatrixRoutingMA(BaseAlpyneZoo):
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
        dispatcher_distance=None,
        max_steps: int = None,
        max_seconds: int = None,
        suppress_initial_reset: int = 0,  # overwrite observation space size if sim cannot be started before initializing spaces
        verbose=False,
        global_client=None,
    ):
        self.fleetsize = fleetsize
        self.max_fleetsize = max_fleetsize

        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.suppress_initial_reset = suppress_initial_reset
        self.verbose = verbose

        self.stepcounter = 0
        self.context = None

        self.possible_agents = [str(agent) for agent in range(fleetsize)]

        self.metadata = dict(is_parallelizable=True)

        self.shuffle()

        if dispatcher is None:
            self.dispatcher = RandDispatcher(dispatcher_distance)
        else:
            self.dispatcher = dispatcher
        self.client = None
        self.started = False
        self.model_path = model_path
        self.startport = startport

        self.config = build_config(config_args, self.fleetsize)
        # Relevant for MA Learning
        self.config.reward_separateAgv = True
        self.config.routingOnNode = True
        self.config.obs_includeNodesInReach = True

        self.masks = {agent: None for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        super().__init__(None, self.possible_agents)

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
        if self.verbose:
            print(f"reset")
        if not self.started:
            self.start()
        self.shuffle()
        self.stepcounter = 0
        self.seed()
        return super().reset(config, seed, return_info, options)

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

    def _catch_nontraining(self, observation: Observation) -> Observation:
        while observation.caller == "Dispatching":
            action = self.dispatcher(observation)
            self.sim.take_action(action)
            self.sim.wait_for_completion()
            observation = self.sim.get_observation()
        return observation

    def _get_observation_space(self, agent) -> spaces.Box:
        """Describe the dimensions and bounds of the observation"""
        if self.suppress_initial_reset == 0:
            if self.observations[agent] is None:
                self.reset()
            obs_sample = self.observe(agent)
            shape = tuple(obs_sample.shape)
        else:
            shape = (self.max_fleetsize * self.suppress_initial_reset,)
        return spaces.Box(low=0, high=1, shape=shape)

    def _get_action_space(self, agent) -> spaces.Space:
        return spaces.Discrete(5)

    def last(self, observe=True) -> "BaseAlpyneZoo.PyObservationType":
        value = super().last(observe)
        self.rewards[self.agent_selection] = 0
        return value

    def _save_observation(self, observation: Observation):
        agent = str(int(observation.caller))
        self.agent_selection = agent
        self.rewards[self.agent_selection] = +observation.rew[0]
        #tea
        #for agent in self.agents:
        #    self.rewards[agent] += observation.rew[0]
        self._cumulative_rewards[self.agent_selection] += observation.rew[0] / 50
        obs_agent = torch.Tensor(observation.obs)
        [self._save_for_agent(obs_agent, str(i), i) for i in range(self.fleetsize)]

    def _save_for_agent(self, obs_complete: torch.Tensor, agent: str, i: int):
        awaiting_orders = obs_complete[i, 1]
        inSystem = obs_complete[i, 0]

        nrows = obs_complete.shape[0] + self.max_fleetsize - self.fleetsize
        ncols = obs_complete.shape[1] + 1
        obs_converted = torch.zeros((nrows, ncols))
        if inSystem:
            obs_converted[0, 0] = 1  # Attention encoding for the AGV to decide for
            obs_converted[0, 1:] = obs_complete[i]
            obs_complete = obs_complete[[x != i for x in range(len(obs_complete))]]
            for src, tgt in enumerate(self.shufflerule):
                obs_converted[tgt + 1, 1:] = obs_complete[src]
            obs_converted[self.max_fleetsize :, 1:] = obs_complete[self.fleetsize :]
        self.observations[agent] = np.array(obs_converted.flatten())
        mask = [
            True,
        ] + [not torch.all(act == 0) for act in obs_converted[0, 9:17].reshape(4, 2)]
        self.masks[agent] = mask

    def _convert_to_action(self, action: "BaseAlpyneEnv.PyActionType", agent) -> Action:
        """Convert the action sent as part of the Gym interface to an Alpyne Action object"""
        action = action if action is not None else 0
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
                    value=int(agent),
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
