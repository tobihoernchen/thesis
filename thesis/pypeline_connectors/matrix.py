import json
import ray.rllib.algorithms.dqn as dqn
from alpyne.data.spaces import Observation
from ..utils.utils import get_config, setup_ray

setup_ray(port=51160)

model_path = "../../models/minimatrix_routing/13_unidirectional_8_10_2023-01-09_12-15-17"#trained_for_pypeline/all_pseudo_mini"
checkpoint = 300
path = "D:/Master/Masterarbeit/thesis"
with open(model_path + "/config.json") as json_file:
    hparams = json.load(json_file)
hparams["n_envs"] = 1
hparams["run_class"] = "pypeline"
config, logger_creator, checkpoint_dir = get_config(path, **hparams)
config["num_gpus"] = 0
trainer = dqn.DQN(config, logger_creator=logger_creator)
trainer.restore(
    model_path + f"/checkpoint_{str(checkpoint).rjust(6, '0')}"
)

env = trainer.workers.local_worker().env.env
env_config = env.config


def get_config(id: str):
    if id in env_config.names():
        return env_config.values()[env_config.names().index(id)]
    return "NOT FOUND"


def get_action(observation, caller, n_nodes, context, statTitles, statValues, time):

    alpyne_obs = Observation(
        obs=observation, caller=caller, n_nodes=n_nodes, networkcontext=context, rew=[0], statTitles = statTitles, statValues = statValues
    )
    agent = env._agent_selection_fn(alpyne_obs)
    print(alpyne_obs)
    if env.agent_behaviors[agent].is_for_learning(alpyne_obs):
        print("wrong")
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
        return env.agent_behaviors[agent].get_action(alpyne_obs, time=time)
