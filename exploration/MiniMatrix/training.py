
import sys
path = "D:/Master/Masterarbeit/thesis"
sys.path.append(path)
from thesis.utils.utils import setup_ray, save, load, Experiment
setup_ray(path = path, unidirectional = True, seed=69)


env_args = dict(
    fleetsize = 4,
    max_fleetsize = 30,    
    pseudo_routing = False,
    pseudo_dispatcher = True,
    #pseudo_dispatcher_distance = 0.2,
    routing_agent_death= False,
    death_on_target = False,
    sim_config = dict(
        dispatch = True,
        routing_ma = True,
        dispatching_ma = True,
        reward_reached_target = 1,
        #reward_reached_target_by_time = True, 
        reward_wrong_target = -0.5,
        reward_removed_for_block = -1, 
        #reward_target_distance = -0.05,
        reward_invalid= -0.1,
        block_timeout = 120,
        #reward_accepted_in_station = 1,
        #reward_declined_in_station = -1,
        reward_geo_operation = 1,
        reward_rework_operation = 0.2,
        reward_respot_operation = 1,
        routing_interval = 2,
        dispatching_interval=360,
        io_quote = 0.95  ,
        availability = 0.9,
        mttr = 5,
    )
)

agv_model = dict(
    model = dict(
        custom_model = "lin_model",
        #custom_action_dist="MAActionDistribution",
        custom_model_config = dict(
            embed_dim=256,
            with_action_mask=False,
            with_agvs=True,
            with_stations = False,
            position_embedd_dim = 0,
            ff_embedd_dim = 4,
            #env_type = "minimatrix",
        )
    )
)
dispatcher_model = dict(
    model = dict(
        custom_model = "lin_model",
        #custom_action_dist="MAActionDistribution",
        custom_model_config = dict(
            embed_dim=128,
            with_action_mask=False,
            with_agvs=True,
            with_stations = True,
            position_embedd_dim = 0,
            ff_embedd_dim = 4,
        )
    )
)

exp = Experiment("minimatrix_routing")
for seed in [42]:
    exp.experiment(
        path = path,
        env_args = env_args, 
        agv_model = agv_model,
        dispatcher_model=dispatcher_model,
        run_name="try_to_scale", 
        env = "matrix",
        algo = "dqn",
        n_intervals =9,
        #train_agv = False,
        backup_interval=50,
        #batch_size=500, #apex + gnn: 50
        seed = seed,
        algo_params = {"gamma":0.98, "exploration_config": {"type": "Curiosity", "sub_exploration": {"type": "StochasticSampling"},}},
        lr = 1e-3,
        #load_agv="../../models/minimatrix_dispatching/04_complete__2_10_2023-01-13_12-09-48/checkpoint_000150/"
    )

exp.keep_training(3)


