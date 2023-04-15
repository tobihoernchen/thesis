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
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy

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

from .double_trainer import DoubleTrainer, TripleTrainer


def seed_all(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)

class Experiment:
    def __init__(self, folder):
        self.trainer = None
        self.folder = folder

    def experiment(
            self, 
            path, 
            env_args, 
            agv_model, 
            dispatcher_model, 
            run_name, 
            algo, 
            env,
            n_intervals, 
            batch_size = 1000, 
            train_agv = True, 
            train_dispatcher = True, 
            backup_interval = 100, 
            seed = 42, 
            load_agv=None, 
            lr = 1e-3, 
            algo_params = {},
            n_envs = 8,
            two_fleets = False,
        ):
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
            type = algo,
            lr = lr,
            algo_params = algo_params,
            n_envs=n_envs,
            two_fleets = two_fleets,
        )
        self.trainer = None
        if algo=="ppo":
            self.trainer =  ppo.PPO(config, logger_creator=logger_creator)
        elif algo == "a2c":
            self.trainer = a2c.A2C(config, logger_creator=logger_creator)
        elif algo == "dqn" or algo == "rainbow":
            self.trainer = dqn.DQN(config, logger_creator=logger_creator)
        elif algo == "apex":
            self.trainer = apex_dqn.ApexDQN(config, logger_creator=logger_creator)
        elif algo == "double" and not two_fleets:
            self.trainer = DoubleTrainer(config, logger_creator=logger_creator)
        elif algo == "double" and  two_fleets:
            self.trainer = TripleTrainer(config, logger_creator=logger_creator)
        if load_agv is not None:
            self.trainer.restore(load_agv)
        self.checkpoint_dir = checkpoint_dir
        self.backup_interval = backup_interval
        for j in range(n_intervals):
            for i in range(backup_interval):
                self.trainer.train()    
            self.trainer.save(checkpoint_dir)

    def keep_training(self, n_intervals):
        for j in range(n_intervals):
            for i in range(self.backup_interval):
                self.trainer.train()    
            self.trainer.save(self.checkpoint_dir)

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
    type = None,
    two_fleets = False,
):
    config = dict()

    if all_ma:
        agv_model["model"]["custom_model_config"]["discrete_action_space"] = True
        dispatcher_model["model"]["custom_model_config"]["discrete_action_space"] = True

    policies = dict()

    policies["agv"] = PolicySpec(
        policy_class = DQNTorchPolicy if type=="double" else None,
        config=agv_model
        )
    if dispatcher_model is not None:
        if not two_fleets:
            policies["dispatcher"] = PolicySpec(
                policy_class = PPOTorchPolicy if type=="double" else None,
                config=dispatcher_model
                )
        else:
            policies["dispatcher1"] = PolicySpec(
                policy_class = PPOTorchPolicy if type=="double" else None,
                config=dispatcher_model
                )
            policies["dispatcher2"] = PolicySpec(
                policy_class = PPOTorchPolicy if type=="double" else None,
                config=dispatcher_model
                )

    policies_to_train = []
    if train_agv:
        policies_to_train.append("agv")
    if dispatcher_model is not None and train_dispatcher:
        if not two_fleets:
            policies_to_train.append("dispatcher")
        else:
            policies_to_train.append("dispatcher1")
            policies_to_train.append("dispatcher2")
    if not two_fleets:
        def pmfn(agent_id, episode, worker, **kwargs):
            if int(agent_id) >= 1000 and int(agent_id) != 2000:
                return "dispatcher"
            else:
                return "agv"
    else:
        def pmfn(agent_id, episode, worker, **kwargs):
            if int(agent_id) >= 1000 and int(agent_id) != 2000:
                return f"dispatcher{int(agent_id) % 2 + 1}"
            else:
                return "agv"

    config["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": pmfn,
        "policies_to_train": policies_to_train,
        "count_steps_by": "agent_steps",
    }
    return config


def config_ppo_training(batch_size=5000, algo_params:dict={}):
    config = {}
    config["train_batch_size"] = batch_size
    config["sgd_minibatch_size"] = algo_params.get("sgd_batch_size",batch_size / 2)
    # config["entropy_coeff"] = 0.01
    config["gamma"] = algo_params.get("gamma", 0.98)
    # config["lambda"] = 0.9
    # config["kl_coeff"] = 0
    # config["lr"] = 3e-5
    # config["lr_schedule"] = [[0, 3e-3], [5000000, 3e-5]]
    # config["vf_loss_coeff"] = 1
    # config["clip_param"] = 0.2
    return config


def config_a2c_training(batch_size=5000, algo_params:dict={}):
    config = {}
    config["train_batch_size"] = batch_size
    config["gamma"] = algo_params.get("gamma", 0.98)
    return config


def config_apex_training(batch_size=5000, algo_params:dict={}):
    config = {}
    config["train_batch_size"] = batch_size
    config["gamma"] = algo_params.get("gamma", 0.98)
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


def config_dqn_training(batch_size=5000, algo_params:dict={}):
    config = {}
    config["train_batch_size"] = batch_size
    config["gamma"] = algo_params.get("gamma", 0.98)
    config["hiddens"] = []
    #New after minimatrix_routing_opt
    config["exploration_config"] = algo_params.get("exploration_config", {
        "epsilon_timesteps": 100000,
        "final_epsilon": 0.02,
        "initial_epsilon": 1.0,
        "type": "EpsilonGreedy"
    })
    return config


def config_rainbow_training(batch_size=5000, algo_params:dict={}):
    config = {}
    config["train_batch_size"] = batch_size
    config["gamma"] = algo_params.get("gamma", 0.98)
    config["n_step"] = 5
    config["num_atoms"] = 5
    config["noisy"] = True
    config["v_min"] = -1
    config["v_max"] = 1
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
    n_envs=8,
    env="minimatrix",
    run_class="default",
    run_name = "-",
    algo_params = {},
    lr = 1e-3,
    two_fleets = False
):
    if type == "ppo":
        config = ppo.DEFAULT_CONFIG.copy()
        add_to_config(config, config_ppo_training(batch_size, algo_params))
    elif type == "a2c":
        config = a2c.A2C_DEFAULT_CONFIG.copy()
        add_to_config(config, config_a2c_training(batch_size, algo_params))
    elif type == "dqn":
        config = dqn.DEFAULT_CONFIG.copy()
        add_to_config(config, config_dqn_training(batch_size, algo_params))
    elif type == "apex":
        config = apex_dqn.APEX_DEFAULT_CONFIG.copy()
        add_to_config(config, config_apex_training(batch_size, algo_params))
    elif type == "rainbow":
        config = dqn.DEFAULT_CONFIG.copy()
        add_to_config(config, config_rainbow_training(batch_size, algo_params))
    elif type == "double":
        config = AlgorithmConfig().to_dict()
        add_to_config(agv_model, config_dqn_training(batch_size, algo_params))
        add_to_config(dispatcher_model, config_ppo_training(batch_size, algo_params))


    config["framework"] = "torch"
    config["callbacks"] = lambda: CustomCallback()
    config["batch_mode"] = "truncate_episodes"  # "complete_episodes"

    #New after minimatrix_routing_opt
    config["metrics_num_episodes_for_smoothing"] = 16

    config["num_gpus"] = 1
    config["num_workers"] = 0
    config["lr"] = lr
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
            type = type,
            two_fleets = two_fleets
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
                algo_params = algo_params,
                two_fleets = two_fleets,
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
