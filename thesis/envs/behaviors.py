from .base_alpyne_zoo import ZooAgentBehavior
from alpyne.data.spaces import Observation, Action
import random
from gym import spaces
import numpy as np
from collections import OrderedDict


class ContextualAgent(ZooAgentBehavior):
    nodes = None
    stations = None
    statistics = None

    def __init__(self) -> None:
        super().__init__()

    def is_for_learning(self, alpyne_obs: Observation) -> bool:
        self._catch_context(alpyne_obs)
        return super().is_for_learning(alpyne_obs)

    def _catch_context(self, alpyne_obs: Observation):
        if (
            "networkcontext" in alpyne_obs.names()
            and alpyne_obs.networkcontext is not None
        ):
            # Node Info
            context = alpyne_obs.networkcontext
            nodecontext = [c for c in context if c[-1] == 0]
            nodes = OrderedDict({i: tuple(n[:2]) for i, n in enumerate(nodecontext)})
            nodes_reversed = OrderedDict({coords: i for i, coords in nodes.items()})
            stations = OrderedDict(
                {i: tuple(n[:2]) for i, n in enumerate(nodecontext) if n[2] == 1}
            )
            ContextualAgent.nodes = nodes
            ContextualAgent.stations = stations

        if "statTitles" in alpyne_obs.names() and "rout_len" in alpyne_obs.statTitles:
            ContextualAgent.statistics = {
                title: value
                for title, value in zip(alpyne_obs.statTitles, alpyne_obs.statValues)
            }

    def _make_action(self, actions, receiver):
        return Action(
            data=[
                {
                    "name": "actions",
                    "type": "INTEGER_ARRAY",
                    "value": list(actions),
                    "unit": None,
                },
                {
                    "name": "receiver",
                    "type": "INTEGER",
                    "value": receiver,
                    "unit": None,
                },
            ]
        )


class RandomStationDispatcher(ContextualAgent):
    def __init__(self, n_actions=1, distance=None) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.distance = distance

    def is_for_learning(self, alpyne_obs: Observation) -> bool:
        super().is_for_learning(alpyne_obs)
        return False

    def _distance(self, a, b):
        return np.sqrt(np.square(a[0] - b[0]) + np.square(a[1] - b[1]))

    def get_action(self, alpyne_obs):
        assert super().stations is not None
        stations = [list(self.stations.keys()) for _ in range(self.n_actions)]
        if self.distance is not None:
            if alpyne_obs.caller == 2001:
                agents = list(range(len(alpyne_obs.obs)))
            else:
                agents = [alpyne_obs.caller % 1000]
            positions = [alpyne_obs.obs[agent][4:6] for agent in agents]
            close_stations = [
                [
                    station_nr
                    for station_nr, station_coord in self.stations.items()
                    if self._distance(position, station_coord) < self.distance
                ]
                for position in positions
            ]
        actions = [
            random.choice(close_stations[i])
            if self.distance is not None and len(close_stations[i]) > 0
            else random.choice(stations[i])
            for i in range(self.n_actions)
        ]
        return self._make_action(actions, alpyne_obs.caller)


class OnlyStandBehavior(ContextualAgent):
    def __init__(self, n_actions=1) -> None:
        super().__init__()
        self.n_actions = n_actions

    def is_for_learning(self, alpyne_obs: Observation) -> bool:
        return False

    def get_action(self, alpyne_obs):
        return self._make_action(
            [
                0,
            ]
            * self.n_actions,
            alpyne_obs.caller,
        )


class TrainingBehavior(ContextualAgent):
    observation_space = None
    action_space = None

    def __init__(self, max_fleetsize, all_ma) -> None:
        super().__init__()
        self.max_fleetsize = max_fleetsize
        self.ma = all_ma

    def is_for_learning(self, alpyne_obs: Observation) -> bool:
        super().is_for_learning(alpyne_obs)
        if not hasattr(self, "n_agv"):
            self._gen_spaces()
        return True

    def _gen_spaces(self):
        self.n_agv = int(ContextualAgent.statistics["n_agv"])
        self.n_stat = int(ContextualAgent.statistics["n_stat"])
        rout_len = int(ContextualAgent.statistics["rout_len"])
        disp_len = int(ContextualAgent.statistics["disp_len"])
        self.action_max = max(self.n_stat, 5)
        self.obs_shape_len = max(rout_len, disp_len)
        # worst case spaces, because RLLIB can't take different space sizes per agent
        if not self.ma:
            TrainingBehavior.action_space = spaces.MultiDiscrete(
                [
                    self.action_max,
                ]
                * self.max_fleetsize
            )
        else:
            TrainingBehavior.action_space = spaces.Discrete(self.action_max)

        spacedict = OrderedDict()
        spacedict["agvs"] = spaces.Box(0, 1, (self.max_fleetsize, self.obs_shape_len))
        spacedict["stat"] = spaces.Box(0, 1, (self.n_stat, self.obs_shape_len))
        spacedict["action_mask"] = (
            spaces.Box(0, 1, (self.max_fleetsize, self.action_max))
            if not self.ma
            else spaces.Box(0, 1, (self.action_max,))
        )
        TrainingBehavior.observation_space = spaces.Dict(spacedict)

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self) -> spaces.Space:
        return self.action_space

    def _state_dict(self, obs):
        include_nodes_in_reach = True
        result = dict(agvs={}, stations={})
        for i, agv in enumerate(obs["agvs"]):
            result["agvs"][str(i)] = d = {}
            d["in_system"] = agv[0]
            d["moving"] = agv[1]
            d["last_node"] = agv[2:4]
            d["next_node"] = agv[4:6]
            hwm = 6
            if not self.dispatching:
                d["target_node"] = agv[hwm : hwm + 2]
                hwm = hwm + 2
            if include_nodes_in_reach:
                d["nodes_in_reach"] = [
                    agv[hwm : hwm + 2],
                    agv[hwm + 2 : hwm + 4],
                    agv[hwm + 4 : hwm + 6],
                    agv[hwm + 6 : hwm + 8],
                ]
                hwm = hwm + 8
            if not self.dispatching:
                d["part_info"] = agv[hwm:]
        for i, station in enumerate(obs["stat"]):
            result["stations"]["station" + str(i)] = d = {}
            d["position"] = station[:2]
            d["state_vec"] = station[2:]
        return result


class SingleAgent(TrainingBehavior):
    def __init__(
        self, max_fleetsize, start_of_nodes_in_reach, dispatching=False
    ) -> None:
        super().__init__(max_fleetsize, all_ma=False)
        self.shufflerules = dict()
        self.dispatching = dispatching
        self.start_of_nodes_in_reach = start_of_nodes_in_reach

    def convert_from_observation(self, alpyne_obs):
        self.shufflerules[alpyne_obs.caller] = srule = random.sample(
            range(self.max_fleetsize), self.n_agv
        )
        reward = alpyne_obs.rew
        in_obs = np.array(alpyne_obs.obs)

        obs_agvs = np.zeros((self.max_fleetsize, self.obs_shape_len))
        in_obs_agvs = in_obs[: self.n_agv]
        obs_agvs[srule, : in_obs_agvs.shape[1]] = in_obs_agvs

        obs_stat = np.zeros((self.n_stat, self.obs_shape_len))
        in_obs_stat = in_obs[self.n_agv :]
        obs_stat[: in_obs_stat.shape[0], : in_obs_stat.shape[1]] = in_obs_stat
        assert not self.ma
        obs_action_mask = np.ones((self.max_fleetsize, self.action_max))
        nodes_in_reach = obs_agvs[
            :, self.start_of_nodes_in_reach : self.start_of_nodes_in_reach + 8
        ]
        nodes_in_reach = (nodes_in_reach.reshape(self.max_fleetsize, 4, 2) != 0).any(
            axis=2
        ) * 1
        obs_action_mask[:, 1:5] = nodes_in_reach
        obs_action_mask[:, 5:] = 0
        obs = OrderedDict()
        obs["agvs"] = obs_agvs
        obs["stat"] = obs_stat
        obs["action_mask"] = obs_action_mask
        return obs, reward, False, {}

    def convert_to_action(self, actions, agent):
        if self.dispatching:
            actions = [
                list(self.stations.keys())[action]
                if action < len(self.stations)
                else list(self.stations.keys())[0]
                for action in actions
            ]
        else:
            actions = [action if action < 5 else 0 for action in actions]
        srule = self.shufflerules[int(agent)]
        reshuffled = [actions[val] for val in srule]
        return self._make_action(reshuffled, int(agent))


class MultiAgent(TrainingBehavior):
    def __init__(
        self,
        max_fleetsize,
        start_of_nodes_in_reach,
        dispatching=False,
        all_ma=False,
        die_on_target=False,
    ) -> None:
        super().__init__(max_fleetsize, all_ma)
        self.shufflerules = dict()
        self.dispatching = dispatching
        self.start_of_nodes_in_reach = start_of_nodes_in_reach
        self.die_on_target = die_on_target

    def convert_from_observation(self, alpyne_obs):
        self.shufflerules[alpyne_obs.caller] = srule = random.sample(
            range(1, self.max_fleetsize), self.n_agv - 1
        )
        reward = alpyne_obs.rew
        in_obs = np.array(alpyne_obs.obs)

        obs_agvs = np.zeros((self.max_fleetsize, self.obs_shape_len))
        caller = alpyne_obs.caller % 1000
        other = list(range(self.n_agv))
        other.remove(caller)
        in_obs_caller = in_obs[caller]
        in_obs_agvs = in_obs[other]
        obs_agvs[0, : in_obs_agvs.shape[1]] = in_obs_caller
        obs_agvs[srule, : in_obs_agvs.shape[1]] = in_obs_agvs

        obs_stat = np.zeros((self.n_stat, self.obs_shape_len))
        in_obs_stat = in_obs[self.n_agv :]
        obs_stat[: in_obs_stat.shape[0], : in_obs_stat.shape[1]] = in_obs_stat

        if self.ma:
            obs_action_mask = np.ones((self.action_max))
        else:
            obs_action_mask = np.ones((self.max_fleetsize, self.action_max))
        nodes_in_reach = obs_agvs[
            0, self.start_of_nodes_in_reach : self.start_of_nodes_in_reach + 8
        ]
        nodes_in_reach = (nodes_in_reach.reshape(4, 2) != 0).any(axis=1) * 1
        if self.ma:
            obs_action_mask[1:5] = nodes_in_reach
        else:
            obs_action_mask[0, 1:5] = nodes_in_reach

        obs = OrderedDict()
        obs["agvs"] = obs_agvs
        obs["stat"] = obs_stat
        obs["action_mask"] = obs_action_mask
        if self.dispatching or not self.die_on_target:
            at_target = False
        else:
            at_target = alpyne_obs.rew >= 5
        return (
            obs,
            reward,
            False,
            {"in_system": in_obs_caller[0] == 1, "at_target": at_target},
        )

    def convert_to_action(self, actions, agent):
        if not isinstance(actions, list):
            action = actions if actions is not None else 0
        else:
            action = actions[0] if actions is not None else 0
        if self.dispatching:
            action = (
                list(self.stations.keys())[action]
                if action < len(self.stations)
                else list(self.stations.keys())[0]
            )
        else:
            action = action if action < 5 else 0
        return self._make_action(
            [
                action,
            ],
            int(agent),
        )
