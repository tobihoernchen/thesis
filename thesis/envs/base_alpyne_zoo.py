from abc import abstractmethod
from typing import List
from typing import Union, Tuple, Dict, Optional

import numpy as np
from gym import spaces

from alpyne.client.model_run import ModelRun
from alpyne.data.spaces import Observation, Action, Configuration

from pettingzoo import AECEnv


class BaseAlpyneZoo(AECEnv):
    """
    An abstract PettingZoo environment.

    """

    # The possible types that the gym Space objects can represent.
    PyObservationType = PyActionType = Union[
        np.ndarray,
        int,
        float,
        Tuple[np.ndarray, int, float, tuple, dict],
        Dict[str, Union[np.ndarray, int, float, tuple, dict]],
    ]

    def __init__(self, sim: ModelRun, agents: List[str]):
        """
        Construct a new environment for the provided sim.

        Note that the configuration passed as part of its creation is what will be used for all episodes.
        If it's desired to have this be changed, the configuration values
        should be assigned to callable values or tuples consisting of (start, stop, step).
        See the `ModelRun` or `Configuration` class documentation for more information.

        :param sim: a created - but not yet started - instance of your model
        :raise ValueError: if the run has been started
        """

        self.agents = list(agents)
        self.agents_original = agents
        # instantiated with first reset call
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: dict() for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.agent_selection = None
        self.observable = False

        self.observation_spaces = {
            agent: self._get_observation_space(agent) for agent in self.agents
        }
        self.action_spaces = {
            agent: self._get_action_space(agent) for agent in self.agents
        }

    def start(self, sim: ModelRun):
        # complain if the sim was already started
        if sim.id:
            raise ValueError("The provided model run should not have been started!")

        self.sim = sim.run()  # submit new run
        self.sim.wait_for_completion()  # wait until start is finished setting up

    def observation_space(self, agent: str) -> spaces.Space:
        """Describe the dimensions and bounds of the observation for the agent

        :param agent: The agent's name"""
        return self.observation_spaces[agent]

    def action_space(self, agent) -> spaces.Space:
        """Describe the dimensions and bounds of the action for the agent

        :param agent: The agent's name"""
        return self.action_spaces[agent]

    def remove_agent(self, agent):
        self.agents.remove(agent)

    def add_agent(self, agent):
        if not agent in self.agents:
            self.agents.append(agent)

    @abstractmethod
    def _get_observation_space(self, agent: str) -> spaces.Space:
        """Describe the dimensions and bounds of the observation for the agent

        :param agent: The agent's name"""
        raise NotImplementedError()

    @abstractmethod
    def _get_action_space(self, agent) -> spaces.Space:
        """Describe the dimensions and bounds of the action for the agent

        :param agent: The agent's name"""
        raise NotImplementedError()

    @abstractmethod
    def _convert_to_action(
        self, action: "BaseAlpyneZoo.PyActionType", agent: str
    ) -> Action:
        """Convert the action sent as part of the Gym interface to an Alpyne Action object

        :param agent: The agent's name"""
        raise NotImplementedError()

    @abstractmethod
    def _save_observation(self, observation: Observation):
        """From the observation, update:
        -self.agent_selection,
        -self.dones
        -self.observations
        -self.rewards
        -self.infos.
        HAS TO REMOVE AGENTS ONCE THEY HAVE FINISHED
        :param observation: Alpyne Observation from ModelRun.get_observation()
        """
        raise NotImplementedError()

    def observe(self, agent: str) -> "BaseAlpyneZoo.PyObservationType":
        """Return the observation that agent currently can make."""
        if not self.observable:
            self.collect()
        return self.observations[agent]

    def _terminal_alternative(self, observation: Observation) -> bool:
        """Optional method to add *extra* terminating conditions"""
        return False

    def _catch_nontraining(self, observation: Observation) -> Observation:
        """Optional method to catch information return from the ModelRun that is irrelevant for Training"""
        return observation

    def step(
        self, action: "BaseAlpyneZoo.PyActionType"
    ) -> Tuple["BaseAlpyneZoo.PyObservationType", float, bool, Optional[dict]]:
        """
        A method required as part of the pettingzoo interface to run one step of the sim.
        Take an action in the sim and advance the sim to the start of the next step.

        :param action: The action to send to the sim (in the type expressed by your action space)
        """
        alpyne_action = self._convert_to_action(action, self.agent_selection)
        self.sim.take_action(alpyne_action)
        self.collect()
        self.observable = True

    def reset(
        self,
        config: Configuration = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
        agents = None,
    ) -> "BaseAlpyneZoo.PyObservationType":
        """
        A method required as part of the pettingzoo interface to revert the sim to the start.

        :param config: The config to start the run with. If None, the sim will use the same configuration object as it was created with.
        """
        if config is not None:
            config.seed = seed if seed is not None else config.seed
            if options is not None:
                for option in options.keys():
                    if option in config.names():
                        config.__setattr__(option, options[option])
        self.agents.clear()
        if agents is None:
            self.agents.extend(self.agents_original)
        else:
            self.agents.extend(agents)
        self.dones = {agent: False for agent in self.agents}

        self.sim.reset(config)
        self.observable = False
        self.collect()
        if return_info:
            return self.observe(self.agent_selection)

    def last(self, observe=True) -> "BaseAlpyneZoo.PyObservationType":
        """
        A method required as part of the pettingzoo interface to gather observation, reward, done and info for the agent that will act next.

        :return: Observation, reward, done and info for the agent that will act next
        """
        if not self.observable:
            self.collect()
        obs = self.observations[self.agent_selection] if observe else None
        reward = self.rewards[self.agent_selection]
        done = self.dones[self.agent_selection]
        info = self.infos[self.agent_selection]
        if done and self.agent_selection in self.agents:
            # All terminate at once
            self.agents.clear()
        return obs, reward, done, info

    def collect(self):
        self.sim.wait_for_completion()
        alpyne_obs = self._catch_nontraining(self.sim.get_observation())
        self._save_observation(alpyne_obs)
        if self.sim.is_terminal() or self._terminal_alternative(alpyne_obs):
            [self.dones.update({agent: True}) for agent in self.agents]
        self.observable = True

    def render(self, mode="human") -> Optional[Union[np.ndarray, str]]:
        """
        A method required as part of the pettingzoo interface to convert the current sim to a useful format
        (e.g., for console printing or animation).

        You may override this method to add on your own custom logic. To see how this is done, see the advanced
        usage of the documentation.
        TODO aadd specific reference in doc to render overriding

        :param mode: the rendering type
        :return: varies based on the mode type
        """
        if mode != "human":
            raise ValueError("Mode not supported: " + mode)

        # lazy implementation; simple printing
        print(
            f"Last status: {self.sim.last_state[0]} | Debug: {self.sim.last_state[1]}"
        )
        return
