from alpyne.data.spaces import Configuration


def build_config(
    config_args: dict, fleetsize: int, seed=None, runmode: int = 4
) -> Configuration:
    return Configuration(
        runmode=runmode,
        fleetsize=fleetsize,
        station_availability=float(
            1 if not "availability" in config_args else config_args["availability"]
        ),
        station_mttr=float(0 if not "mttr" in config_args else config_args["mttr"]),
        station_ioQuote=float(
            1 if not "ioquote" in config_args else config_args["ioquote"]
        ),
        reward_acceptedInStation=float(
            0
            if not "reward_acceptance" in config_args
            else config_args["reward_acceptance"]
        ),
        reward_removedForBlock=float(
            0 if not "reward_block" in config_args else config_args["reward_block"]
        ),
        reward_blocking=float(
            0
            if not "reward_blocking" in config_args
            else config_args["reward_blocking"]
        ),
        reward_geoOperation=float(
            0 if not "reward_geo" in config_args else config_args["reward_geo"]
        ),
        reward_respotOperation=float(
            0 if not "reward_respot" in config_args else config_args["reward_respot"]
        ),
        reward_reworkOperation=float(
            0 if not "reward_rework" in config_args else config_args["reward_rework"]
        ),
        reward_partCompleted=float(
            0
            if not "reward_completion" in config_args
            else config_args["reward_completion"]
        ),
        reward_reachedTarget=float(
            0 if not "reward_target" in config_args else config_args["reward_target"]
        ),
        reward_targetDistance=float(
            0
            if not "reward_distance" in config_args
            else config_args["reward_distance"]
        ),
        reward_separateAgv=False
        if not "reward_separateAgv" in config_args
        else config_args["reward_separateAgv"],
        reward_invalid=float(
            0
            if not "reward_invalid" in config_args
            else config_args["reward_invalid"]
        ),
        obs_includeNodesInReach=False
        if not "includeNodesInReach" in config_args
        else config_args["includeNodesInReach"],
        obs_coordinates=True
        if not "coordinates" in config_args
        else config_args["coordinates"],
        obsdisp_includeAgvTarget=False
        if not "obsdisp_includeAgvTarget" in config_args
        else config_args["obsdisp_includeAgvTarget"],
        obsrout_includePartInfo=False
        if not "obsrout_includePartInfo" in config_args
        else config_args["obsrout_includePartInfo"],
        pypelineName="" if not "pyname" in config_args else config_args["pyname"],
        pypelinePath="" if not "pypath" in config_args else config_args["pypath"],
        routinginterval=float(
            2
            if not "routinginterval" in config_args
            else config_args["routinginterval"]
        ),
        dispatchinginterval=float(
            35
            if not "dispatchinginterval" in config_args
            else config_args["dispatchinginterval"]
        ),
        routingOnNode=False
        if not "routingOnNode" in config_args
        else config_args["routingOnNode"],
        withCollisions=True
        if not "withCollisions" in config_args
        else config_args["withCollisions"],
        blockTimeout=int(
            60 if not "blockTimeout" in config_args else config_args["blockTimeout"]
        ),
        seed=0 if seed is None else seed,  # will be overwitten with every reset
    )
