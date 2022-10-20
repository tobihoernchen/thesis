import os
import tempfile
from datetime import datetime
import torch
import random
import numpy.random

import ray
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents import ppo, dqn
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger

from thesis.policies.simplified_attention_module import register_attention_model
from thesis.policies.just_lin import register_lin_model
from thesis.envs.matrix_routing_zoo import MatrixRoutingMA
from thesis.envs.matrix_dispatching_zoo import MatrixDispatchingMA
from thesis.envs.matrix_dispatching_zoo_death import MatrixDispatchingMA_Death
from thesis.envs.matrix_zoo import MatrixMA
from thesis.utils.callbacks import CustomCallback


def setup_ray(path="../..", env="Routing"):
    torch.manual_seed(42)
    random.seed(42)
    numpy.random.seed(42)
    os.environ["PYTHONPATH"] = path
    ray.shutdown()
    if env == "Routing":
        setup_matrix_routing_for_ray()
    if env == "Complete":
        setup_matrix_complete_for_ray()
    if env == "Dispatching":
        setup_matrix_dispatching_for_ray()
    if env == "Death":
        setup_matrix_death_for_ray()
    register_attention_model()
    register_lin_model()
    ray.init(ignore_reinit_error=True, include_dashboard=True)


def rllib_ppo_config(
    fleetsize,
    max_fleetsize,
    config_args,
    n_envs=4,
    n_stations=5,
    dispatching=False,
    with_dispatcher=False,
    lin_model=False,
):
    config = ppo.DEFAULT_CONFIG.copy()
    env_args = dict(
        dispatcher="stat" if with_dispatcher else None,
        max_seconds=3600,
        fleetsize=fleetsize,
        max_fleetsize=max_fleetsize,
        config_args=config_args,
    )

    agvconfig = dict(
        fleetsize=max_fleetsize,
        embed_dim=32,
        n_stations=n_stations,
        depth=6,
        with_action_mask=True,
        with_stations=False,
    )

    dispconfig = dict(
        fleetsize=max_fleetsize,
        embed_dim=16,
        n_stations=n_stations,
        depth=4,
        with_action_mask=True,
        with_agvs=False,
    )

    if dispatching:
        policies = {
            "agv": PolicySpec(
                config={
                    "model": dict(custom_model_config=agvconfig),
                }
            ),
            "agv1": PolicySpec(
                config={
                    "model": dict(custom_model_config=agvconfig),
                }
            ),
            "dispatcher": PolicySpec(
                config={
                    "model": dict(custom_model_config=dispconfig),
                    "gamma": 0.8,
                    "lambda": 0.8,
                }
            ),
            "dispatcher1": PolicySpec(
                config={
                    "model": dict(custom_model_config=dispconfig),
                    "gamma": 0.8,
                    "lambda": 0.8,
                }
            ),
        }

    else:
        policies = {
            "agv": PolicySpec(config={"model": dict(custom_model_config=agvconfig)})
        }

    config["framework"] = "torch"
    config["callbacks"] = lambda: CustomCallback()

    config["num_gpus"] = 1
    config["num_workers"] = 0
    config["num_envs_per_worker"] = n_envs
    config["train_batch_size"] = 2000 * 5
    config["sgd_minibatch_size"] = 2000
    config["entropy_coeff"] = 0.01
    config["gamma"] = 0.98
    config["lambda"] = 0.95
    config["kl_coeff"] = 0
    config["lr"] = 3e-5
    config["lr_schedule"] = [[0, 3e-4], [10000000, 3e-8]]
    config["vf_loss_coeff"] = 0.5
    config["clip_param"] = 0.2

    config["batch_mode"] = "complete_episodes"

    config["env"] = "matrix"
    config["env_config"] = env_args

    def pmfn(agent_id, episode, worker, **kwargs):
        alias = worker.env.env.current_alias
        if agent_id.endswith("Dispatching"):
            if agent_id.startswith(str(alias[0])):
                return "dispatcher1"
            return "dispatcher"
        else:
            # if agent_id.startswith(str(alias[1])):
            #     return "agv1"
            return "agv"

    config["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": pmfn,
        "policies_to_train": [
            "agv",
            "dispatcher1",
        ],
        "count_steps_by": "agent_steps",
    }

    config["model"].update(
        custom_model="attention_model" if not lin_model else "lin_model",
        vf_share_layers=True,
    )

    return config


def setup_matrix_routing_for_ray(verbose=False):
    env_fn = lambda config: MatrixRoutingMA(
        model_path="D://Master/Masterarbeit/thesis/envs/MiniMatrix.zip",
        verbose=verbose,
        **config
    )
    register_env("matrix", lambda config: PettingZooEnv(env_fn(config)))


def setup_matrix_dispatching_for_ray(verbose=False):
    env_fn = lambda config: MatrixDispatchingMA(
        model_path="D://Master/Masterarbeit/thesis/envs/MiniMatrix.zip",
        verbose=verbose,
        **config
    )
    register_env("matrix", lambda config: PettingZooEnv(env_fn(config)))


def setup_matrix_death_for_ray(verbose=False):
    env_fn = lambda config: MatrixDispatchingMA_Death(
        model_path="D://Master/Masterarbeit/thesis/envs/MiniMatrix.zip",
        verbose=verbose,
        **config
    )
    register_env("matrix", lambda config: PettingZooEnv(env_fn(config)))


def setup_matrix_complete_for_ray(verbose=False):
    env_fn = lambda config: MatrixMA(
        model_path="D://Master/Masterarbeit/thesis/envs/MiniMatrix.zip",
        verbose=verbose,
        **config
    )
    register_env("matrix", lambda config: PettingZooEnv(env_fn(config)))


def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def save(trainer, name: str, path: str):
    torch.save(trainer.workers.local_worker().get_policy(name).model.state_dict(), path)


def load(trainer, name, path):
    trainer.workers.local_worker().get_policy(name).model.load_state_dict(
        torch.load(path)
    )
