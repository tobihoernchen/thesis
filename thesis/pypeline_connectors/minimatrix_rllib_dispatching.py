import json
import torch

import ray.rllib.agents.ppo as ppo
from alpyne.data.spaces import Observation
from ..utils.rllib_utils import rllib_ppo_config, setup_ray

setup_ray(env="Death")

model_path = "../../models/Default/6-10-23_10-10_03_26/checkpoint_000180/checkpoint-180"
hparams_path = "../../models/Default/6-10-23_10-10_03_26.json"

with open(hparams_path) as json_file:
    hparams = json.load(json_file)
fleetsize = hparams["fleetsize"]
max_fleetsize = hparams["max_fleetsize"]
config = hparams["env_args"]

trainer = ppo.PPOTrainer(
    rllib_ppo_config(
        fleetsize,
        max_fleetsize,
        config,
        n_envs=1,
        n_stations=5,
        dispatching=True,
        # with_dispatcher=True,
        lin_model=True,
        startport=51140,
    )
)
trainer.restore(model_path)

env = trainer.workers.local_worker().env.env
config = env.config


def get_config(id: str):
    if id in config.names():
        return config.values()[config.names().index(id)]
    return "NOT FOUND"


def get_action(observation, caller, n_nodes, context):
    alpyneobs = Observation(
        obs=observation, caller=caller, n_nodes=n_nodes, networkcontext=context, rew=[0]
    )
    if caller.endswith("Dispatching"):
        policy = "dispatcher"
    else:
        policy = "agv"

    env._save_observation(alpyneobs)
    action = trainer.get_policy(policy).compute_single_action(
        env.last()[0], explore=False
    )[0]

    return env._convert_to_action(action, caller)
