from alpyne.data.spaces import Configuration


def build_config(
    config_args: dict, fleetsize: int, seed=None, runmode: int = 1
) -> Configuration:
    return Configuration(
        ### SIM Params
        runmode=runmode,
        fleetsize=fleetsize,
        seed=0 if seed is None else seed,  # will be overwitten with every reset
        dispatch=True if not "dispatch" in config_args else config_args["dispatch"],
        routing_ma=True
        if not "routing_ma" in config_args
        else config_args["routing_ma"],
        dispatching_ma=False
        if not "dispatching_ma" in config_args
        else config_args["dispatching_ma"],
        routing_interval=float(
            2
            if not "routing_interval" in config_args
            else config_args["routing_interval"]
        ),
        dispatching_interval=float(
            35
            if not "dispatching_interval" in config_args
            else config_args["dispatching_interval"]
        ),
        coordinates=True
        if not "coordinates" in config_args
        else config_args["coordinates"],
        with_collisions=True
        if not "with_collisions" in config_args
        else config_args["with_collisions"],
        block_timeout=int(
            60 if not "block_timeout" in config_args else config_args["block_timeout"]
        ),
        remove_all_blocked=True
        if not "remove_all_blocked" in config_args
        else config_args["remove_all_blocked"],
        ### AGV Params
        reward_target_distance=float(
            0
            if not "reward_target_distance" in config_args
            else config_args["reward_target_distance"]
        ),
        reward_reached_target=float(
            0
            if not "reward_reached_target" in config_args
            else config_args["reward_reached_target"]
        ),
        reward_wrong_target=float(
            0
            if not "reward_wrong_target" in config_args
            else config_args["reward_wrong_target"]
        ),
        reward_removed_for_block=float(
            0
            if not "reward_removed_for_block" in config_args
            else config_args["reward_removed_for_block"]
        ),
        reward_invalid=float(
            0 if not "reward_invalid" in config_args else config_args["reward_invalid"]
        ),
        obs_include_nodes_in_reach=False
        if not "obs_include_nodes_in_reach" in config_args
        else config_args["obs_include_nodes_in_reach"],
        obs_include_agv_target=False
        if not "obs_include_agv_target" in config_args
        else config_args["obs_include_agv_target"],
        obs_include_part_info=False
        if not "obs_include_part_info" in config_args
        else config_args["obs_include_part_info"],
        ### STATION Params
        reward_declined_in_station=float(
            0
            if not "reward_declined_in_station" in config_args
            else config_args["reward_declined_in_station"]
        ),
        reward_accepted_in_station=float(
            0
            if not "reward_accepted_in_station" in config_args
            else config_args["reward_accepted_in_station"]
        ),
        reward_geo_operation=float(
            0
            if not "reward_geo_operation" in config_args
            else config_args["reward_geo_operation"]
        ),
        reward_rework_operation=float(
            0
            if not "reward_rework_operation" in config_args
            else config_args["reward_rework_operation"]
        ),
        reward_respot_operation=float(
            0
            if not "reward_respot_operation" in config_args
            else config_args["reward_respot_operation"]
        ),
        reward_part_completed=float(
            0
            if not "reward_part_completed" in config_args
            else config_args["reward_part_completed"]
        ),
        station_availability=float(
            1 if not "availability" in config_args else config_args["availability"]
        ),
        station_mttr=float(0 if not "mttr" in config_args else config_args["mttr"]),
        station_io_quote=float(
            1
            if not "station_io_quote" in config_args
            else config_args["station_io_quote"]
        ),
    )
