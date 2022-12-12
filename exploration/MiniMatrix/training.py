# %%
from ray.rllib.algorithms import ppo, a3c, dqn, apex_dqn
from ray import tune
import numpy as np
import torch

import sys

sys.path.append("../..")
from thesis.utils.utils import get_config, setup_ray, save, load

path = "D:/Master/Masterarbeit/thesis"
setup_ray(path=path, unidirectional=False)

# %%
env_args = dict(
    fleetsize=2,
    max_fleetsize=10,
    pseudo_routing=False,
    pseudo_dispatcher=True,
    # pseudo_dispatcher_distance = 0.3,
    routing_agent_death=True,
    death_on_target=True,
    sim_config=dict(
        dispatch=True,
        routing_ma=True,
        dispatching_ma=True,
        reward_reached_target=10,
        # reward_reached_target_by_time = True,
        reward_wrong_target=-0.5,
        reward_removed_for_block=-1,
        # reward_target_distance = -0.05,
        reward_invalid=-0.2,
        block_timeout=20,
        reward_accepted_in_station=1,
        reward_declined_in_station=-1,
        dispatching_interval=360,
        io_quote=0.99,
        availability=0.95,
        mttr=5 * 60,
    ),
)

# %%
agv_model = dict(
    model=dict(
        custom_model="attn_model",
        # custom_action_dist="MAActionDistribution",
        custom_model_config=dict(
            env_type="minimatrix",
            embed_dim=16,
            with_action_mask=True,
            # with_agvs=True,
            with_stations=False,
            # n_convolutions = 4
        ),
    )
)
dispatcher_model = dict(
    model=dict(
        custom_model="lin_model",
        # custom_action_dist="MAActionDistribution",
        custom_model_config=dict(
            embed_dim=16,
            with_action_mask=False,
            with_agvs=True,
            with_stations=True,
        ),
    )
)

# %%
config, logger_creator, checkpoint_dir = get_config(
    path=path,
    batch_size=1000,
    env_args=env_args,
    agv_model=agv_model,
    train_agv=True,
    dispatcher_model=dispatcher_model,
    train_dispatcher=True,
    env="minimatrix",
    run_class="comparison",
    type="ppo",
)
trainer = ppo.PPO(config, logger_creator=logger_creator)
# trainer = a3c.A3CTrainer(config, logger_creator=logger_creator)
# trainer = dqn.DQN(config, logger_creator=logger_creator)
##trainer = apex_dqn.ApexDQN(config, logger_creator=logger_creator)

# %%
# trainer.save(checkpoint_dir)

# %%
# trainer.restore("../../models/comparison/6_30_2022-11-14_17-44-48/checkpoint_000300/checkpoint-300")

# %%
for j in range(10):
    for i in range(100):
        trainer.train()
    trainer.save(checkpoint_dir)

# %%
# save(trainer, "agv", "../../models/trained")
