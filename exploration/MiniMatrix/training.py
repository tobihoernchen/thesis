# %%
from ray.rllib.agents import ppo, dqn
from ray import tune
import numpy as np
import torch

import sys

sys.path.append("./thesis")
from thesis.utils.utils import get_config, setup_ray, save, load

setup_ray()

# %%
env_args = dict(
    fleetsize=6,
    max_fleetsize=10,
    pseudo_dispatcher=True,
    routing_agent_death=True,
    sim_config=dict(
        dispatch=True,
        routing_ma=True,
        dispatching_ma=True,
        reward_reached_target=1,
        reward_wrong_target=-0.3,
        reward_removed_for_block=-1,
        reward_target_distance=0.2,
        reward_invalid=-0.1,
        block_timeout=20,
        reward_accepted_in_station=1,
        reward_declined_in_station=-1,
        dispatching_interval=240,
        io_quote=0.99,
        availability=0.95,
        mttr=5 * 60,
    ),
)

# %%
agv_model = dict(
    model=dict(
        custom_model="lin_model",
        custom_action_dist="MAActionDistribution",
        custom_model_config=dict(
            embed_dim=16,
            with_action_mask=True,
            with_agvs=True,
            with_stations=False,
        ),
    )
)
dispatcher_model = dict(
    model=dict(
        custom_model="lin_model",
        custom_model_config=dict(
            embed_dim=16,
            with_action_mask=True,
            with_agvs=True,
            with_stations=True,
        ),
    )
)

# %%
config, logger_creator, checkpoint_dir = get_config(
    batch_size=5000,
    env_args=env_args,
    agv_model=agv_model,
    train_agv=True,
    dispatcher_model=dispatcher_model,
    train_dispatcher=True,
)
trainer = ppo.PPOTrainer(config, logger_creator=logger_creator)

# %%
for j in range(500):
    for i in range(20):
        trainer.train()
    trainer.save(checkpoint_dir)
