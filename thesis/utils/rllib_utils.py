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
from thesis.envs.matrix_routing_zoo import MatrixRoutingMA
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
    register_attention_model()
    ray.init(ignore_reinit_error=True, include_dashboard=True)


def rllib_dqn_config(
    fleetsize,
    max_fleetsize,
    config_args,
    max_seconds=3600,
    lr=None,
    n_envs=4,
    lin_model=None,
    use_attention=None,
):

    config = dqn.DEFAULT_CONFIG.copy()
    env_args = dict(
        max_seconds=max_seconds,
        fleetsize=fleetsize,
        max_fleetsize=max_fleetsize,
        config_args=config_args,
    )
    config["framework"] = "torch"

    config["num_gpus"] = 1
    config["num_workers"] = 0
    config["num_envs_per_worker"] = n_envs
    if lr != None:
        config["lr"] = lr

    config["env"] = "matrix"
    config["env_config"] = env_args
    config["multiagent"] = {
        "policies": {
            "agv": PolicySpec(),
        },
        "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "agv",
    }

    if lin_model is None:
        config["model"].update(custom_model="attention_model")
    else:
        config["model"].update(fcnet_hiddens=lin_model, use_attention=use_attention)

    return config


def rllib_ppo_config(
    fleetsize,
    max_fleetsize,
    config_args,
    max_seconds=3600,
    lr=None,
    entropy_coeff=None,
    batchsize=None,
    n_envs=4,
    lin_model=None,
    use_attention=None,
    gamma=None,
    n_stations=5,
):
    config = ppo.DEFAULT_CONFIG.copy()
    env_args = dict(
        max_seconds=max_seconds,
        fleetsize=fleetsize,
        max_fleetsize=max_fleetsize,
        config_args=config_args,
        with_action_masks=True,
    )
    config["framework"] = "torch"
    config["callbacks"] = lambda: CustomCallback()

    config["num_gpus"] = 1
    config["num_workers"] = 0
    config["num_envs_per_worker"] = n_envs
    if batchsize is not None:
        config["train_batch_size"] = batchsize * 10
        config["sgd_minibatch_size"] = batchsize
    if lr is not None:
        config["lr"] = lr
    if entropy_coeff is not None:
        config["entropy_coeff"] = entropy_coeff
    if gamma is not None:
        config["gamma"] = gamma
    config["lambda"] = 0.98
    config["kl_coeff"] = 0#.1

    config["env"] = "matrix"
    config["env_config"] = env_args
    config["multiagent"] = {
        "policies": {
            "agv": PolicySpec(),
        },
        "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "agv",
    }

    if lin_model is None:
        config["model"].update(
            custom_model="attention_model",
            custom_model_config=dict(
                fleetsize=max_fleetsize,
                embed_dim=64,
                n_stations=n_stations,
                depth=4,
                with_action_mask=True,
            ),
        )
    else:
        config["model"].update(fcnet_hiddens=lin_model, use_attention=use_attention)

    return config


def setup_matrix_routing_for_ray(verbose=False):
    env_fn = lambda config: MatrixRoutingMA(
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
