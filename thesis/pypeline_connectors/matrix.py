import json
import ray.rllib.algorithms.dqn as dqn
from alpyne.data.spaces import Observation
from ..utils.utils import get_config, setup_ray

setup_ray(port=51160, unidirectional=False)

model_path = "../../models/matrix_routing/primal_8_20_2023-02-11_23-21-50"#trained_for_pypeline/new_all_pseudo"#minimatrix_dispatching/06_mat_rout__4_30_2023-01-13_18-43-09"
checkpoint = 650
checkpoint_path = model_path + f"/checkpoint_{str(checkpoint).rjust(6, '0')}"
path = "D:/Master/Masterarbeit/thesis"
with open(model_path + "/config.json") as json_file:
    hparams = json.load(json_file)
hparams["n_envs"] = 1
hparams["run_class"] = "pypeline"
config, logger_creator, checkpoint_dir = get_config(path, **hparams)
config["num_gpus"] = 1
trainer = dqn.DQN(config, logger_creator=logger_creator)
trainer.restore(checkpoint_path)

env = trainer.workers.local_worker().env.env
env_config = env.config


def get_config(id: str):
    if id in env_config.names():
        return env_config.values()[env_config.names().index(id)]
    return "NOT FOUND"


def get_action(observation, caller, n_nodes, context, statTitles, statValues, time):
    alpyne_obs = Observation(
        obs=observation, caller=caller, n_nodes=n_nodes, networkcontext=context, rew=0, statTitles = statTitles, statValues = statValues
    )
    agent = env._agent_selection_fn(alpyne_obs)
    if env.agent_behaviors[agent].is_for_learning(alpyne_obs):
        env._save_observation(
            alpyne_obs,
            agent,
            *env.agent_behaviors[agent].convert_from_observation(alpyne_obs),
        )
        print(agent)
        print(config["multiagent"]["policy_mapping_fn"](agent, None, None))
        action = trainer.compute_single_action(env.last()[0], explore=False, policy_id=config["multiagent"]["policy_mapping_fn"](agent, None, None))

        return env.agent_behaviors[env.agent_selection].convert_to_action(
            action, caller
        )

    else:
        return env.agent_behaviors[agent].get_action(alpyne_obs, time=time)
