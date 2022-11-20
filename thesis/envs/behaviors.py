from .base_alpyne_zoo import ZooAgentBehavior
from alpyne.data.spaces import Observation, Action
import random
from gym import spaces
import numpy as np
from collections import OrderedDict
from typing import List


class HiveContext:
    def __init__(self) -> None:
        self.nodes = None
        self.stations = None
        self.statistics = None
        self.paths = None

    def set_statistics(self, stat_dict):
        self.statistics = stat_dict


class ContextualAgent(ZooAgentBehavior):
    def __init__(self, hive: HiveContext) -> None:
        super().__init__()
        self.hive = hive

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
            nodecontext = [c for c in context if c[-2] == 0]
            nodes = OrderedDict({i: tuple(n[:2]) for i, n in enumerate(nodecontext)})
            nodes_reversed = OrderedDict({coords: i for i, coords in nodes.items()})
            pathcontext = [c for c in context if c[-2] != 0]
            paths = []
            for x1, y1, x2, y2, bi in pathcontext:
                node1 = nodes_reversed[(x1, y1)]
                node2 = nodes_reversed[(x2, y2)]
                paths.append((node1, node2))
                if bi == 1:
                    paths.append((node2, node1))

            stations = OrderedDict(
                {i: tuple(n[:2]) for i, n in enumerate(nodecontext) if n[2] == 1}
            )
            self.hive.nodes = nodes
            self.hive.stations = stations
            self.hive.paths = paths

        if "statTitles" in alpyne_obs.names() and "rout_len" in alpyne_obs.statTitles:
            self.hive.set_statistics(
                {
                    title: value
                    for title, value in zip(
                        alpyne_obs.statTitles, alpyne_obs.statValues
                    )
                }
            )

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
    def __init__(self, hive: HiveContext, n_actions=1, distance=None) -> None:
        super().__init__(hive)
        self.n_actions = n_actions
        self.distance = distance

    def is_for_learning(self, alpyne_obs: Observation) -> bool:
        super().is_for_learning(alpyne_obs)
        return False

    def _distance(self, a, b):
        return np.sqrt(np.square(a[0] - b[0]) + np.square(a[1] - b[1]))

    def get_action(self, alpyne_obs, time):
        assert self.hive.stations is not None
        stations = [list(self.hive.stations.keys()) for _ in range(self.n_actions)]
        if self.distance is not None:
            if alpyne_obs.caller == 2001:
                agents = list(range(len(alpyne_obs.obs)))
            else:
                agents = [alpyne_obs.caller % 1000]
            positions = [alpyne_obs.obs[agent][4:6] for agent in agents]
            close_stations = [
                [
                    station_nr
                    for station_nr, station_coord in self.hive.stations.items()
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
    def __init__(self, hive: HiveContext, n_actions=1) -> None:
        super().__init__(hive)
        self.n_actions = n_actions

    def is_for_learning(self, alpyne_obs: Observation) -> bool:
        return False

    def get_action(self, alpyne_obs, time):
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

    def __init__(self, hive: HiveContext, max_fleetsize, all_ma) -> None:
        super().__init__(hive)
        self.max_fleetsize = max_fleetsize
        self.ma = all_ma

    def is_for_learning(self, alpyne_obs: Observation) -> bool:
        super().is_for_learning(alpyne_obs)
        if not hasattr(self, "n_agv"):
            self._gen_spaces()
        return True

    def _gen_spaces(self):
        self.n_agv = int(self.hive.statistics["n_agv"])
        self.n_stat = int(self.hive.statistics["n_stat"])
        rout_len = int(self.hive.statistics["rout_len"])
        disp_len = int(self.hive.statistics["disp_len"])
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
        self,
        hive: HiveContext,
        max_fleetsize,
        start_of_nodes_in_reach,
        dispatching=False,
    ) -> None:
        super().__init__(hive, max_fleetsize, all_ma=False)
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
                list(self.hive.stations.keys())[action]
                if action < len(self.hive.stations)
                else list(self.hive.stations.keys())[0]
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
        hive: HiveContext,
        max_fleetsize,
        start_of_nodes_in_reach,
        dispatching=False,
        all_ma=False,
        die_on_target=False,
    ) -> None:
        super().__init__(hive, max_fleetsize, all_ma)
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
                list(self.hive.stations.keys())[action]
                if action < len(self.hive.stations)
                else list(self.hive.stations.keys())[0]
            )
        else:
            action = action if action < 5 else 0
        return self._make_action(
            [
                action,
            ],
            int(agent),
        )


class Arc:
    def __init__(self, i, node1, inode1, node2, inode2, maxima) -> None:
        self.i = i
        self.start = inode1
        self.end = inode2
        self.length = (
            np.sqrt(np.sum(np.square(np.multiply(np.subtract(node1, node2), maxima))))
            * 1.2
        )
        self.travel_time = self.length
        self.geo_conflicts = []
        self.blocked_time_windows = []
        self.free_time_windows = [(0, np.inf)]
        self.out_arcs = []

    def add_block_window(self, start, end, who):
        self.blocked_time_windows.append((start, end, who))
        self.calc_free_time_windows()

    def clear_blocks(self, who):
        self.blocked_time_windows = [
            window for window in self.blocked_time_windows if window[2] != who
        ]
        self.calc_free_time_windows()

    def time_window_intersect(self, window1, window2):
        if window1[0] < window2[0] and not window1[1] <= window2[0]:
            return True
        if window1[1] > window2[1] and not window1[0] >= window2[1]:
            return True
        return False

    def calc_free_time_windows(self):
        self.free_time_windows = []
        self.blocked_time_windows.sort(key=lambda w: w[0])
        start = 0
        for window in self.blocked_time_windows:
            if window[0] - start > 0.1:
                self.free_time_windows.append((start, window[0]))
            start = window[1]
        self.free_time_windows.append((start, np.inf))

    def get_first_free_timeslot(self, start_time):
        blocks_after_start = [
            window for window in self.blocked_time_windows if window[1] > start_time
        ]
        possible_time_window_starts = [start_time] + [
            block[1] for block in blocks_after_start
        ]
        possible_time_windows = [
            (start, start + self.travel_time) for start in possible_time_window_starts
        ]
        for window in possible_time_windows:
            if not any(
                [
                    self.time_window_intersect(window, other)
                    for other in self.blocked_time_windows
                ]
            ):
                break
        return window


class Label:
    def __init__(self, arc, distance, start, end, predecessor) -> None:
        self.arc: Arc = arc
        self.distance = distance
        self.start = start
        self.end = end
        self.predecessor = predecessor

    def dominates(self, other):
        if (
            self.distance <= other.distance
            and self.start <= other.start
            and self.end >= other.end
        ):
            return True


class CollisionFreeRoutingHive(HiveContext):
    def __init__(self) -> None:
        super().__init__()
        self.arcs = None
        self.maxima = None

    def set_statistics(self, stat_dict):
        if "maxX" in stat_dict.keys() and "maxY" in stat_dict.keys():
            self.maxima = (stat_dict["maxX"] / 10, stat_dict["maxY"] / 10)
        return super().set_statistics(stat_dict)

    def clear_blocks(self, who):
        for arc in self.arcs.values():
            arc.clear_blocks(who)

    def preprocess(self):
        self.arcs = {
            i: Arc(
                i,
                node1=self.nodes[arc[0]],
                inode1=arc[0],
                node2=self.nodes[arc[1]],
                inode2=arc[1],
                maxima=self.maxima,
            )
            for i, arc in enumerate(self.paths)
        }
        for iarc1, arc1 in self.arcs.items():
            for iarc2, arc2 in self.arcs.items():
                if arc1.start == arc2.end and arc1.end == arc2.start:
                    arc1.geo_conflicts.append(arc2)
                elif arc1.end == arc2.start:
                    arc1.out_arcs.append(arc2)

    def compute_route(
        self,
        start_node_coords,
        start_time,
        target_node_coords,
        possible_nodes_coords,
        who,
    ):
        start_node = None
        target_node = None
        possible_nodes = []
        for i, node in self.nodes.items():
            if node == start_node_coords:
                start_node = i
            if node == target_node_coords:
                target_node = i
            if node in possible_nodes_coords:
                possible_nodes.append(i)
        H: List[Label] = []
        start_arcs = [
            arc
            for arc in self.arcs.values()
            if arc.start == start_node and arc.end in possible_nodes
        ]
        for arc in start_arcs:
            H.append(Label(arc, 0, start_time, np.inf, None))
        while len(H) > 0:
            H.sort(key=lambda l: l.distance)
            label = H[0]
            if label.arc.start == target_node:
                break
            for window in label.arc.free_time_windows:
                possible_arrival_start = (
                    max(window[0], label.start) + label.arc.travel_time
                )
                possible_arrival_end = min(window[1], label.end)
                if possible_arrival_start < possible_arrival_end:
                    wait_time = max(0, window[0] - label.start)
                    new_distance = label.distance + label.arc.length + wait_time
                    for next_arc in label.arc.out_arcs:
                        new_label = Label(
                            next_arc,
                            new_distance,
                            possible_arrival_start,
                            possible_arrival_end,
                            label,
                        )
                        if not any(
                            [l.dominates(new_label) for l in H if l.arc == next_arc]
                        ):
                            H.append(new_label)
            H.pop(0)
        if len(H) == 0:
            return None
        else:
            path = []
            label.end = label.start + label.arc.length
            last_start = label.start
            label = label.predecessor
            while label is not None:
                label.end = last_start
                label.start = last_start = label.end - label.arc.length
                path.append([self.nodes[label.arc.end], label.start])
                label.arc.add_block_window(label.start, label.end, who)
                for arc in label.arc.geo_conflicts:
                    arc.add_block_window(label.start, label.end, who)
                label = label.predecessor
            path.reverse()
            return path


class CollisionFreeRouting(ContextualAgent):
    def __init__(self, hive: CollisionFreeRoutingHive) -> None:
        super().__init__(hive)
        self.current_route = None

    def is_for_learning(self, alpyne_obs: Observation) -> bool:
        super()._catch_context(alpyne_obs)
        if self.hive.arcs is None and self.hive.paths is not None:
            self.hive.preprocess()
        return False

    def get_action(self, alpyne_obs, time):
        caller = alpyne_obs.caller
        obs = alpyne_obs.obs
        next_node = tuple(obs[caller][4:6])
        target_node = tuple(obs[caller][6:8])
        possible_nodes = [(obs[caller][i], obs[caller][i + 1]) for i in range(8, 16, 2)]
        if (
            self.current_route is None
            or len(self.current_route) == 0
            or target_node != self.current_route[-1][0]
        ):
            if self.current_route is not None and len(self.current_route) > 0:
                self.hive.clear_blocks(caller)
            self.current_route = self.hive.compute_route(
                next_node, time, target_node, possible_nodes, caller
            )
        if (
            self.current_route is not None
            and len(self.current_route) > 0
            and self.current_route[0][1] <= time
            and self.current_route[0][0] in possible_nodes
        ):
            action = possible_nodes.index(self.current_route[0][0]) + 1
            self.current_route.pop(0)
            return self._make_action([action], caller)
        else:
            return self._make_action([0], caller)
