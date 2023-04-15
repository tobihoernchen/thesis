# %%
import sys
sys.path.append("D:/Master/Masterarbeit/thesis")
from thesis.utils.utils import setup_ray, save, load, Experiment
path = "D:/Master/Masterarbeit/thesis"
setup_ray(path = path, unidirectional = True, seed=69)

# %%
env_args = dict(
    fleetsize = 4,
    max_fleetsize = 20,    
    pseudo_routing = True,
    pseudo_dispatcher = False,
    pseudo_dispatcher_clever = False,
    #pseudo_dispatcher_distance = 0.3,
    routing_agent_death= False,
    death_on_target = False,
    transform_dispatching_partobs = True,
    direction_reward = 0,
    sim_config = dict(
        dispatch = True,
        routing_ma = True,
        dispatching_ma = True,
        reward_reached_target = 0,
        #reward_reached_target_by_time = True, 
        reward_wrong_target = 0,
        reward_removed_for_block = 0, 
        reward_target_distance = 0,
        reward_invalid= 0,
        reward_duration = -0.5,
        block_timeout = 120,
        station_separate = False,
        reward_accepted_in_station = 2,
        reward_declined_in_station = -1,
        #reward_part_completed = 5,
        #reward_geo_operation=1,
        #reward_rework_operation=1,
        #reward_respot_operation=1,
        reward_reduce = -0.02,
        reward_balance = -5,
        routing_interval = 2,
        dispatching_interval=30,
        io_quote = 0.9  ,
        availability = 0.9,
        mttr = 5,
        #fixed_fleets = True,
    )
)

# %%
agv_model = dict(
    model = dict(
        custom_model = "gnn_model",
        #custom_action_dist="MAActionDistribution",
        custom_model_config = dict(
            embed_dim=16,
            with_action_mask=False,
            with_agvs=True,
            with_stations = False,
            position_embedd_dim = 0,
            ff_embedd_dim = 4,
            env_type = "matrix",
            n_convolutions = 2
        )
    )
)
dispatcher_model = dict(
    model = dict(
        custom_model = "lin_model",
        #custom_action_dist="MAActionDistribution",
        custom_model_config = dict(
            embed_dim=32,
            with_action_mask=True,
            with_agvs=True,
            with_stations = True,
            position_embedd_dim = 0,
            ff_embedd_dim = 4,
        )
    )
)

# %%
exp = Experiment("matrix_dispatching")
for seed in [44]:
    exp.experiment(
        path = path,
        env_args = env_args, 
        agv_model = agv_model,
        dispatcher_model=dispatcher_model,
        run_name="09_rew_balance_two_disp", 
        env = "matrix",
        algo = "ppo",
        n_intervals =3,
        train_agv = False,
        backup_interval=20,
        batch_size=100, #apex + gnn: 50
        seed = seed,
        algo_params = {"gamma":0.9,},#"grad_clip": 1,  "exploration_config":{"warmup_timesteps": 0,"epsilon_timesteps": 200000,"final_epsilon": 0.02,"initial_epsilon": 1.0,"type": "EpsilonGreedy"}},# "exploration_config": {"type": "Curiosity", "sub_exploration": {"type": "StochasticSampling"}, "eta": 0.1}},
        lr = 1e-4,
        #load_agv="../../models/matrix_dispatching/07_rew_conf_process_4_20_2023-03-20_23-05-40/checkpoint_000060",
        #two_fleets = True,
        
    )

# %%
#load(exp.trainer, "agv", "../../models/trained")

# %%
save(exp.trainer, "dispatcher", "../../models/trained_disp_new")

# %%
exp.keep_training(4)

# %%
#exp.keep_training(10)


