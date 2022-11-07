import os
import tempfile
from datetime import datetime
import torch
import random
import numpy.random
import json
import time

import ray
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents import ppo, dqn
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger

# from thesis.policies.simplified_attention_module import register_attention_model
from thesis.policies.ma_lin_net import register_lin_model
from thesis.envs.matrix import Matrix
from thesis.utils.callbacks import CustomCallback
from thesis.policies.ma_action_dist import register_ma_action_dist


def setup_ray(path="../.."):
    torch.manual_seed(42)
    random.seed(42)
    numpy.random.seed(42)
    os.environ["PYTHONPATH"] = path
    ray.shutdown()
    setup_minimatrix_for_ray()
    setup_matrix_for_ray()
    # register_attention_model()
    register_lin_model()
    register_ma_action_dist()
    ray.init(ignore_reinit_error=True, include_dashboard=True)


def config_ma_policies(
    agv_model, train_agv=True, dispatcher_model=None, train_dispatcher=True
):
    config = dict()

    policies = dict()
    policies["agv"] = PolicySpec(config=agv_model)
    if dispatcher_model is not None:
        policies["dispatcher"] = PolicySpec(config=dispatcher_model)

    policies_to_train = []
    if train_agv:
        policies_to_train.append("agv")
    if dispatcher_model is not None and train_dispatcher:
        policies_to_train.append("dispatcher")

    def pmfn(agent_id, episode, worker, **kwargs):
        if int(agent_id) >= 1000 and int(agent_id) != 2000:
            return "dispatcher"
        else:
            return "agv"

    config["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": pmfn,
        "policies_to_train": policies_to_train,
        "count_steps_by": "agent_steps",
    }
    return config


def config_ppo_training(batch_size=5000, sgd_batch_size=None):
    config = {}
    config["train_batch_size"] = batch_size
    config["sgd_minibatch_size"] = 5000 if sgd_batch_size is None else sgd_batch_size
    config["entropy_coeff"] = 0.01
    config["gamma"] = 0.98
    config["lambda"] = 0.95
    config["kl_coeff"] = 0
    config["lr"] = 3e-5
    config["lr_schedule"] = [[0, 3e-3], [5000000, 3e-5]]
    config["vf_loss_coeff"] = 1
    config["clip_param"] = 0.2
    return config


def add_to_config(config, to_add: dict):
    for k, v in to_add.items():
        config[k] = v


def get_config(
    env_args,
    agv_model,
    train_agv=True,
    dispatcher_model=None,
    train_dispatcher=True,
    batch_size=2500,
    type="ppo",
    n_envs=4,
    env="minimatrix",
    run_class="default",
):
    if type == "ppo":
        config = ppo.DEFAULT_CONFIG.copy()
        add_to_config(config, config_ppo_training(batch_size))

    config["framework"] = "torch"
    config["callbacks"] = lambda: CustomCallback()
    config["batch_mode"] = "truncate_episodes"  # "complete_episodes"

    config["num_gpus"] = 1
    config["num_workers"] = 0
    config["num_envs_per_worker"] = n_envs

    add_to_config(
        config,
        config_ma_policies(
            agv_model=agv_model,
            train_agv=train_agv,
            dispatcher_model=dispatcher_model,
            train_dispatcher=train_dispatcher,
        ),
    )

    config["env"] = env
    config["env_config"] = env_args

    models_dir = f"../../models/{run_class}"
    logs_dir = f"../../logs/{run_class}"
    run_name = f"{env_args['fleetsize']}_{env_args['max_fleetsize']}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    if not os.path.exists(f"{models_dir}/{run_name}"):
        os.makedirs(f"{models_dir}/{run_name}")
    with open(f"{models_dir}/{run_name}/config.json", "w") as outfile:
        json.dump(
            dict(
                env_args=env_args,
                agv_model=agv_model,
                train_agv=train_agv,
                dispatcher_model=dispatcher_model,
                train_dispatcher=train_dispatcher,
                type=type,
                batch_size=batch_size,
                n_envs=n_envs,
                env=env,
                run_class=run_class,
            ),
            outfile,
            indent=3,
        )

    return config, custom_log_creator(logs_dir, run_name), f"{models_dir}/{run_name}"


def setup_minimatrix_for_ray():
    env_fn = lambda config: Matrix(
        model_path= "../../envs/MiniMatrix.zip",
        max_seconds=60 * 60,
        **config,
    )
    register_env("minimatrix", lambda config: PettingZooEnv(env_fn(config)))


def setup_matrix_for_ray():
    env_fn = lambda config: Matrix(
        model_path="../../envs/Matrix.zip",
        max_seconds=60 * 60,
        **config,
    )
    register_env("matrix", lambda config: PettingZooEnv(env_fn(config)))


def custom_log_creator(custom_path, logdir_prefix):
    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def save(trainer, policy_name: str, path: str):
    torch.save(
        trainer.workers.local_worker().get_policy(policy_name).model.state_dict(), path
    )


def load(trainer, policy_name, path):
    trainer.workers.local_worker().get_policy(policy_name).model.load_state_dict(
        torch.load(path)
    )
