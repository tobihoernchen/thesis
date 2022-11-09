import json
import torch

import ray.rllib.agents.ppo as ppo
from alpyne.data.spaces import Observation
from ..utils.utils import get_config, setup_ray

setup_ray()

model_path = "../../models/Default/2_30_2022-11-07_22-32-06"
checkpoint = 100

with open(model_path + "/config.json") as json_file:
    hparams = json.load(json_file)
hparams["n_envs"] = 1
hparams["run_class"] = "pypeline"
hparams["env_args"]["startport"] = 51160
config, logger_creator, checkpoint_dir = get_config(**hparams)
config["num_gpus"] = 0
trainer = ppo.PPOTrainer(config, logger_creator=logger_creator)
trainer.restore(
    model_path + f"/checkpoint_{str(checkpoint).rjust(6, '0')}/checkpoint-{checkpoint}"
)

env = trainer.workers.local_worker().env.env
env_config = env.config


def get_config(id: str):
    if id in env_config.names():
        return env_config.values()[env_config.names().index(id)]
    return "NOT FOUND"


def get_action(observation, caller, n_nodes, context):
    alpyne_obs = Observation(
        obs=observation, caller=caller, n_nodes=n_nodes, networkcontext=context, rew=[0]
    )
    agent = env._agent_selection_fn(alpyne_obs)
    if env.agent_behaviors[agent].is_for_learning(alpyne_obs):
        env._save_observation(
            alpyne_obs,
            agent,
            *env.agent_behaviors[agent].convert_from_observation(alpyne_obs),
        )
        action = trainer.get_policy(
            config["multiagent"]["policy_mapping_fn"](agent, None, None)
        ).compute_single_action(env.last()[0], explore=False)[0]
        return env.agent_behaviors[env.agent_selection].convert_to_action(
        action, caller
    )

    else:
        return env.agent_behaviors[agent].get_action(alpyne_obs)
    