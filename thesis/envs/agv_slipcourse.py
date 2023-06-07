from cgitb import reset
from alpyne.client.alpyne_client import AlpyneClient
from alpyne.client.abstract import BaseAlpyneEnv
from alpyne.data.spaces import Configuration, Observation, Action
import gym
from gym import spaces
from typing import Union, Tuple, Dict, Optional, NoReturn


class AgvSlipCourse(BaseAlpyneEnv):
    """
    Gym-Adapter for the AGV Slipcourse-Environment (../../envs/AGV_SlipCourse)
    model_path: path to the AnyLogic-model to connect to
    max_steps: number of timesteps to run until an episode is finished
    """
    def __init__(self, model_path, max_steps):
        self.client = AlpyneClient(model_path, port=51153)
        config = Configuration(inSlipChance=0.3)
        self.run = self.client.create_reinforcement_learning(config)
        super(AgvSlipCourse, self).__init__(self.run)
        self.reset()
        self.max_steps = max_steps

    def reset(self) -> "BaseAlpyneEnv.PyObservationType":
        self.stepcounter = 0
        return super().reset()

    def step(
        self, action: "BaseAlpyneEnv.PyActionType"
    ) -> Tuple["BaseAlpyneEnv.PyObservationType", float, bool, Optional[dict]]:
        self.stepcounter += 1
        return super().step(action)

    def _get_observation_space(self) -> spaces.Space:
        """Describe the dimensions and bounds of the observation"""
        return spaces.Box(
            low=0, high=1, shape=(len(self.sim.get_observation().values()[0]),)
        )

    def _convert_from_observation(
        self, observation: Observation
    ) -> "BaseAlpyneEnv.PyObservationType":
        """Convert your Observation object to the format expected by Gym"""
        return observation.values()[0]

    def _get_action_space(self) -> spaces.Space:
        return spaces.Discrete(4)

    def _convert_to_action(self, action: "BaseAlpyneEnv.PyActionType") -> Action:
        """Convert the action sent as part of the Gym interface to an Alpyne Action object"""
        return Action(inAction=int(action))

    def _calc_reward(self, observation: Observation) -> float:
        """Evaluate the performance of the last action based on the current observation"""
        return observation.values()[1] if self.stepcounter != self.max_steps else -10

    def _terminal_alternative(self, observation: Observation) -> bool:
        """Optional method to add *extra* terminating conditions"""
        return self.stepcounter >= self.max_steps
