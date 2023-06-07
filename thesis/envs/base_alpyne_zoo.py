from abc import abstractmethod
from typing import List
from typing import Union, Tuple, Dict, Optional

import numpy as np
from gym import spaces

from alpyne.client.model_run import ModelRun
from alpyne.data.spaces import Observation, Action, Configuration

from pettingzoo import AECEnv


class ZooAgentBehavior:
    """
    Baseclass to describe the behavior of an agent. Any custom behavior should inherit from this. 
    A BaseAlpyneZoo-environment has one of these behaviors for every agent. It can be used for learning agents, agents that just use some static behavior or anything in between. 
    """
    def __init__(self) -> None:
        pass

    def is_for_learning(self, alpyne_obs) -> bool:
        """
        Will be called when the corresponding agent has to act. 
        If True: Delegate to gym-inteface for learning
        If False: Choose the action by get_action and go on with the simulation. 
        """
        return True

    @abstractmethod
    def get_action(self, alpyne_obs, time):
        """
        Calculate the action absed on the alpyne observation object and the current time.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """
        Describe the agent's observation space
        """
        raise NotImplementedError()

    @abstractmethod
    def get_action_space(self) -> spaces.Space:
        """
        Describe the agent's action space
        """
        raise NotImplementedError()

    @abstractmethod
    def convert_to_action(
        self, action: "BaseAlpyneZoo.PyActionType", agent: str
    ) -> Action:
        """
        Convert the action from the learning algorithm / get_action to an alypne action object
        """
        raise NotImplementedError()

    @abstractmethod
    def convert_from_observation(self, alpyne_obs):
        """
        Convert the alpyne observation object to an observation compatible with get_observation_space
        """
        raise NotImplementedError()


class ZooAgent:
    """
    
    """
    def __init__(
        self, name: str, behavior: ZooAgentBehavior, does_training: bool = True
    ) -> None:
        self.name = name
        self.behavior = behavior
        self.does_training = does_training


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

    def __init__(self, agent_behaviors = None):
        """
        Construct a new environment for the provided sim.

        Note that the configuration passed as part of its creation is what will be used for all episodes.
        If it's desired to have this be changed, the configuration values
        should be assigned to callable values or tuples consisting of (start, stop, step).
        See the `ModelRun` or `Configuration` class documentation for more information.

        :param sim: a created - but not yet started - instance of your model
        :raise ValueError: if the run has been started
        """
        self._initialize_cache(agents=agent_behaviors)

        self.agent_selection = None
        self.observable = False

    @abstractmethod
    def _agent_selection_fn(self, alpyne_obs):
        raise NotImplementedError

    @abstractmethod
    def get_agents(self) -> List[ZooAgent]:
        raise NotImplementedError

    def reset(
        self,
        config: Configuration = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
        agents=None,
        dont_collect = False,
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
        self._initialize_cache()
        self.sim.wait_for_completion()
        self.sim.reset(config)
        self.observable = False
        if not dont_collect:
            self._collect()
            if return_info:
                return self.observe(self.agent_selection)

    def step(
        self, action: "BaseAlpyneZoo.PyActionType"
    ) -> Tuple["BaseAlpyneZoo.PyObservationType", float, bool, Optional[dict]]:
        """
        A method required as part of the pettingzoo interface to run one step of the sim.
        Take an action in the sim and advance the sim to the start of the next step.

        :param action: The action to send to the sim (in the type expressed by your action space)
        """
        alpyne_action = self.agent_behaviors[self.agent_selection].convert_to_action(
            action, self.agent_selection
        )
        self.sim.take_action(alpyne_action)
        self._collect()

    def last(self, observe=True) -> "BaseAlpyneZoo.PyObservationType":
        """
        A method required as part of the pettingzoo interface to gather observation, reward, done and info for the agent that will act next.

        :return: Observation, reward, done and info for the agent that will act next
        """
        if not self.observable:
            self._collect()
        obs = self.observations[self.agent_selection] if observe else None
        reward = self.rewards[self.agent_selection]
        done = self.dones[self.agent_selection]
        info = self.infos[self.agent_selection]
        if all([self.dones[agent] for agent in self.agents]):
            # All terminate at once
            self.agents.clear()
        return obs, reward, done, info

    def observe(self, agent: str) -> "BaseAlpyneZoo.PyObservationType":
        """Return the observation that agent currently can make."""
        if not self.observable:
            self._collect()
        return self.observations[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.agent_behaviors[agent].get_observation_space()

    def action_space(self, agent) -> spaces.Space:
        return self.agent_behaviors[agent].get_action_space()

    def remove_agent(self, agent):
        self.agents.remove(agent)

    def add_agent(self, agent: ZooAgent):
        if not agent.name in self.agents:
            self.agents.append(agent.name)
            self.dones[agent.name] = False
            self.infos[agent.name] = dict()
            self.rewards[agent.name] = 0
            self.observations[agent.name] = None
            self.agent_behaviors[agent.name] = agent.behavior

    def _collect(self):
        while True:
            self.sim.wait_for_completion()
            alpyne_obs = self.sim.get_observation()
            agent = self._agent_selection_fn(alpyne_obs)
            if self.agent_behaviors[agent].is_for_learning(alpyne_obs):
                self._save_observation(
                    alpyne_obs,
                    agent,
                    *self.agent_behaviors[agent].convert_from_observation(alpyne_obs)
                )
                break
            else:
                action = self.agent_behaviors[agent].get_action(
                    alpyne_obs, time=self.run.get_state()[1]["model_time"]
                )
                self.sim.take_action(action)

        if self.sim.is_terminal() or self._terminal_alternative(alpyne_obs):
            [self.dones.update({agent: True}) for agent in self.agents]
        self.observable = True

    def _save_observation(self, alpyne_obs, agent, obs, reward, done, info):
        self.agent_selection = agent
        self.observations[agent] = obs
        self.rewards[agent] = reward
        self.dones[agent] = done
        self.infos[agent] = info

    def _terminal_alternative(self, observation: Observation) -> bool:
        """Optional method to add *extra* terminating conditions"""
        return False

    def _initialize_cache(self, agents: List[ZooAgent] = None):
        if agents is None:
            agents = self.get_agents()
        self.agents = [agent.name for agent in agents if agent.does_training]
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: dict() for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.agent_behaviors = {agent.name: agent.behavior for agent in agents}

    def _start(self, sim: ModelRun):
        # complain if the sim was already started
        if sim.id:
            raise ValueError("The provided model run should not have been started!")

        self.sim = sim.run()  # submit new run
        self.sim.wait_for_completion()  # wait until start is finished setting up
