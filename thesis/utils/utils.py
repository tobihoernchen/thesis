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
from ray.rllib.algorithms import ppo, a2c, dqn, apex_dqn
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger

# from thesis.policies.simplified_attention_module import register_attention_model
from thesis.policies.ma_lin_net import register_lin_model
from thesis.policies.ma_attention_net import register_attn_model
from thesis.policies.sa_attention_net import register_sa_attn_model
from thesis.policies.ma_gnn_routing import register_gnn_model
from thesis.policies.sa_plain_net import register_plain_model
from thesis.policies.attentive_gnn_routing import register_attn_gnn_model
from thesis.envs.matrix import Matrix
from thesis.utils.callbacks import CustomCallback
from thesis.policies.ma_action_dist import register_ma_action_dist


def seed_all(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)

class Experiment:
    def __init__(self, folder):
        self.trainer = None
        self.folder = folder

    def experiment(self, path, env_args, agv_model, dispatcher_model, run_name, algo, env, n_intervals, batch_size = 1000, train_agv = True, train_dispatcher = True, backup_interval = 100, seed = 42, load_agv=None):
        seed_all(seed)
        config, logger_creator, checkpoint_dir = get_config(
            path = path,
            batch_size=batch_size,
            env_args = env_args, 
            agv_model = agv_model,
            train_agv = train_agv,
            dispatcher_model=dispatcher_model, 
            train_dispatcher=train_dispatcher,
            env = env,
            run_class=self.folder,
            run_name = run_name,
            type = algo
        )
        if algo=="ppo":
            self.trainer =  ppo.PPO(config, logger_creator=logger_creator)
        elif algo == "a3c":
            self.trainer = a2c.A2C(config, logger_creator=logger_creator)
        elif algo == "dqn" or algo == "rainbow":
            self.trainer = dqn.DQN(config, logger_creator=logger_creator)
        elif algo == "apex":
            self.trainer = apex_dqn.ApexDQN(config, logger_creator=logger_creator)
        if load_agv is not None:
            self.trainer.restore(load_agv)
        for j in range(n_intervals):
            for i in range(backup_interval):
                self.trainer.train()    
            self.trainer.save(checkpoint_dir)

def setup_ray(path="../..", unidirectional=False, port=None, seed =42):
    seed_all(seed)
    os.environ["PYTHONPATH"] = path
    ray.shutdown()
    setup_minimatrix_for_ray(path, unidirectional=unidirectional, port=port)
    setup_matrix_for_ray(path, unidirectional=unidirectional, port=port)
    # register_attention_model()
    register_lin_model()
    register_attn_model()
    register_sa_attn_model()
    register_ma_action_dist()
    register_gnn_model()
    register_attn_gnn_model()
    register_plain_model()
    ray.init(ignore_reinit_error=True, include_dashboard=True, num_gpus = 1)


def config_ma_policies(
    agv_model,
    train_agv=True,
    dispatcher_model=None,
    train_dispatcher=True,
    all_ma=False,
):
    config = dict()

    if all_ma:
        agv_model["model"]["custom_model_config"]["discrete_action_space"] = True
        dispatcher_model["model"]["custom_model_config"]["discrete_action_space"] = True

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
    config["sgd_minibatch_size"] = (
        batch_size / 2 if sgd_batch_size is None else sgd_batch_size
    )
    # config["entropy_coeff"] = 0.01
    # config["gamma"] = 0.9
    # config["lambda"] = 0.9
    # config["kl_coeff"] = 0
    # config["lr"] = 3e-5
    # config["lr_schedule"] = [[0, 3e-3], [5000000, 3e-5]]
    # config["vf_loss_coeff"] = 1
    # config["clip_param"] = 0.2
    return config


def config_a2c_training(batch_size=5000):
    config = {}
    config["train_batch_size"] = batch_size
    config["gamma"] = 0.98
    config["lr"] = 3e-5
    return config


def config_apex_training(batch_size=5000):
    config = {}
    config["train_batch_size"] = batch_size
    config["gamma"] = 0.98
    config["lr"] = 3e-5
    config["replay_buffer_config"] ={
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 10000,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-6,
            "replay_sequence_length": 1,
            "worker_side_prioritization": False,
        }
    return config


def config_dqn_training(batch_size=5000):
    config = {}
    config["train_batch_size"] = batch_size
    config["gamma"] = 0.98
    config["lr"] = 3e-5
    config["hiddens"] = []
    return config


def config_rainbow_training(batch_size=5000):
    config = {}
    config["train_batch_size"] = batch_size
    config["gamma"] = 0.98
    config["n_step"] = 5
    config["num_atoms"] = 5
    config["noisy"] = True
    config["v_min"] = -1
    config["v_max"] = 1
    config["lr"] = 3e-5
    return config


def add_to_config(config, to_add: dict):
    for k, v in to_add.items():
        config[k] = v


def get_config(
    path,
    env_args,
    agv_model,
    train_agv=True,
    dispatcher_model=None,
    train_dispatcher=True,
    batch_size=5000,
    type="ppo",
    n_envs=4,
    env="minimatrix",
    run_class="default",
    run_name = "-",
):
    if type == "ppo":
        config = ppo.DEFAULT_CONFIG.copy()
        add_to_config(config, config_ppo_training(batch_size))
    elif type == "a3c":
        config = a2c.DEFAULT_CONFIG.copy()
        add_to_config(config, config_a2c_training(batch_size))
    elif type == "dqn":
        config = dqn.DEFAULT_CONFIG.copy()
        add_to_config(config, config_dqn_training(batch_size))
    elif type == "apex":
        config = apex_dqn.APEX_DEFAULT_CONFIG.copy()
        add_to_config(config, config_apex_training(batch_size))
    elif type == "rainbow":
        config = dqn.DEFAULT_CONFIG.copy()
        add_to_config(config, config_rainbow_training(batch_size))

    config["framework"] = "torch"
    config["callbacks"] = lambda: CustomCallback()
    config["batch_mode"] = "truncate_episodes"  # "complete_episodes"

    config["num_gpus"] = 1
    config["num_workers"] = 0
    config["lr"] = 1e-3
    config["log_level"] = "ERROR"
    config["num_envs_per_worker"] = n_envs

    all_ma = env_args["sim_config"]["routing_ma"] and (
        env_args["sim_config"]["dispatching_ma"]
        or not env_args["sim_config"]["dispatch"]
    )

    add_to_config(
        config,
        config_ma_policies(
            agv_model=agv_model,
            train_agv=train_agv,
            dispatcher_model=dispatcher_model,
            train_dispatcher=train_dispatcher,
            all_ma=all_ma,
        ),
    )

    config["env"] = env
    config["env_config"] = env_args

    models_dir = f"{path}/models/{run_class}"
    logs_dir = f"{path}/logs/{run_class}"
    run_name = f"{run_name}_{env_args['fleetsize']}_{env_args['max_fleetsize']}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
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


def setup_minimatrix_for_ray(path, unidirectional=False, port=None):
    path = path + (
        "/envs/MiniMatrix.zip"
        if not unidirectional
        else "/envs/MiniMatrix_unidirectional.zip"
    )
    env_fn = lambda config: Matrix(
        startport=51150 + config.worker_index if port is None else port,
        model_path=path,
        max_seconds=60 * 60,
        **config,
    )
    register_env("minimatrix", lambda config: PettingZooEnv(env_fn(config)))


def setup_matrix_for_ray(path, unidirectional=False, port=None):
    path = path + (
        "/envs/Matrix.zip" if not unidirectional else "/envs/Matrix_unidirectional.zip"
    )
    env_fn = lambda config: Matrix(
        startport=51150 + config.worker_index if port is None else port,
        model_path=path,
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
