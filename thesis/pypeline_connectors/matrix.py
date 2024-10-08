import json
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.dqn as dqn
from ..utils.double_trainer import DoubleTrainer, TripleTrainer
from alpyne.data.spaces import Observation
from ..utils.utils import get_config, setup_ray

### SETUP ###
setup_ray(port=51160, unidirectional=True)
pseudo = True
model_path = "../../models/matrix_routing/09_rew_block_-20_8_20_2023-05-01_22-56-02"#minimatrix_dispatching/06_mat_rout__4_30_2023-01-13_18-43-09"
checkpoint = 400
trainer_cls = dqn.DQN
fleetsize = 16#None
### END SETUP ###


checkpoint_path = model_path + f"/checkpoint_{str(checkpoint).rjust(6, '0')}"
path = "C:/Users/Wegma/Documents/Masterarbeit/thesis/"
with open(model_path + "/config.json") as json_file:
    hparams = json.load(json_file)
hparams["n_envs"] = 1
hparams["run_class"] = "pypeline"
if fleetsize is not None:
    hparams["env_args"]["fleetsize"] = fleetsize
    hparams["env_args"]["fleetsize_upper"] = None
if pseudo:
    hparams["env_args"]["max_seconds"] = 10
    hparams["env_args"]["pseudo_dispatcher"] = True
    hparams["env_args"]["pseudo_dispatcher_clever"] = True
    #hparams["env_args"]["pseudo_dispatcher_rule"] = "shortest_quene"
    hparams["env_args"]["pseudo_routing"] = True
    
config, logger_creator, checkpoint_dir = get_config(path, **hparams)
config["num_gpus"] = 1
trainer = trainer_cls(config, logger_creator=logger_creator)
if not pseudo:
    trainer.restore(checkpoint_path)

env = trainer.workers.local_worker().env.env
if pseudo:  
    env.reset(options = {"dont_collect": True})
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
