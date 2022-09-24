import json
import torch

import ray.rllib.agents.ppo as ppo
from alpyne.data.spaces import Observation, Action

from ..envs.matrix_zoo import MatrixMA
from ..utils.build_config import build_config
from ..utils.utils import RewardCheck


env = MatrixMA(
    model_path="../../envs/MiniMatrix.zip",
    startport=51144,
    fleetsize=6,
    max_fleetsize=10,
    with_action_masks=True,
    config_args=dict(
        reward_block=-1,
        # reward_acceptance = 0.5,
        # reward_geo = 0.05,
        # reward_respot = 0.01,
        # reward_rework = 0.05,
        reward_completion=1,
        reward_separateAgv=True,
        routingOnNode=True,
        withCollisions=True,
    ),
)

chk = RewardCheck(env, "build")

config = env.config


def get_config(id: str):
    if id in config.names():
        return config.values()[config.names().index(id)]
    return "NOT FOUND"


def get_action(observation, caller, n_nodes, context):
    alpyneobs = Observation(
        obs=observation, caller=caller, n_nodes=n_nodes, networkcontext=context, rew=[0]
    )

    env._save_observation(alpyneobs)
    action = chk.obs2action(env.last()[0])[0]

    return env._convert_to_action(action, caller)
