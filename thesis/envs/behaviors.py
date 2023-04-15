from .base_alpyne_zoo import ZooAgentBehavior
from .part_variants import MatrixPart
from alpyne.data.spaces import Observation, Action
import random
from gym import spaces
import numpy as np
from collections import OrderedDict
from typing import List
import matplotlib.pyplot as plt
from ray.rllib.policy.policy import Policy


class HiveContext:
    def __init__(self) -> None:
        self.nodes = None
        self.stations = None
        self.statistics = None
        self.paths = None
        self.policies = {}

    def set_statistics(self, stat_dict):
        self.statistics = stat_dict

    def draw(self):
        for inode1, inode2 in self.paths:
            node1 = self.nodes[inode1]
            node2 = self.nodes[inode2]
            plt.arrow(node1[0], -node1[1], node2[0]-node1[0], node1[1]-node2[1], length_includes_head = True, head_width = 0.008)
        plt.show()

    def add_policy(self, policy_path):
        if not policy_path in self.policies.keys():
            self.policies[policy_path] = Policy.from_checkpoint(policy_path)
        return self.policies[policy_path]

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

        if "statTitles" in alpyne_obs.names() and "maxX" in alpyne_obs.statTitles:
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

    def part_obs_to_dict(self, part_obs, env_type):
        if not any(part_obs):
            part_info = None
        else:
            if env_type == "matrix":
                part_info = {
                    "variant": ["vb_1", "vb_2", "hc_1", "hc_2"][list(part_obs[:4]).index(1)],
                    "state": ["g01", "g12", "g23", "g34", "g45", "g56", "g6d", "finalState"][list(part_obs[4:12]).index(1)],
                    "respots": [part_obs[12 + 2 * i] for i in range(10)],
                    "nio": [part_obs[13 + 2 * i] for i in range(10)]
                }
            else:
                part_info = {
                    "variant": ["vb_1", "vb_2"][list(part_obs[:2]).index(1)],
                    "state": ["g01", "g12", "g2d", "finalState"][list(part_obs[2:8]).index(1)],
                    "respots": [part_obs[8 + 2 * i] for i in range(2)],
                    "nio": [part_obs[9 + 2 * i] for i in range(2)]
                }   
        return part_info


class ApplyPolicyAgent(ContextualAgent):
    def __init__(self, hive: HiveContext, policy_path, dummy_learning_agent) -> None:
        super().__init__(hive)
        self.policy:Policy = self.hive.add_policy(policy_path)
        self.dummy_learning_agent:TrainingBehavior = dummy_learning_agent

    def is_for_learning(self, alpyne_obs: Observation) -> bool:
        super().is_for_learning(alpyne_obs)
        self.dummy_learning_agent.is_for_learning(alpyne_obs)
        return False

    def get_action(self, alpyne_obs, time):
        obs, rew, done, info = self.dummy_learning_agent.convert_from_observation(alpyne_obs)
        action = self.policy.compute_single_action(obs, explore=False)
        return self._make_action([action[0]], alpyne_obs.caller)


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
        assert self.hive.stations is not None, "Hive did not yet receive any context data."
        if alpyne_obs.caller == 2001:
            station_list = self.hive.stations
        else:
            agent_position = tuple(alpyne_obs.obs[alpyne_obs.caller % 1000][4:6])
            station_list = {
                k: v for k, v in self.hive.stations.items() if v != agent_position
            }
        stations = [list(station_list.keys()) for _ in range(self.n_actions)]
        if self.distance is not None:
            if alpyne_obs.caller == 2001:
                agents = list(range(len(alpyne_obs.obs)))
            else:
                agents = [alpyne_obs.caller % 1000]
            positions = [alpyne_obs.obs[agent][4:6] for agent in agents]
            close_stations = [
                [
                    station_nr
                    for station_nr, station_coord in station_list.items()
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


class CleverMatrixDispatcher(ContextualAgent):
    def __init__(self, hive: HiveContext, *args, **kwargs) -> None:
        super().__init__(hive)
        self.part=MatrixPart()
        self.env_type="matrix"
        self.assigned = None



    def is_for_learning(self, alpyne_obs: Observation) -> bool:
        super().is_for_learning(alpyne_obs)
        return False

    def get_action(self, alpyne_obs, time):
        relevant_obs = alpyne_obs.obs[alpyne_obs.caller - 1000]
        part_obs = np.array(relevant_obs[18:])
        part_info = self.part_obs_to_dict(part_obs, self.env_type)
        part_info = self.part.translate(part_info) if part_info is not None else None
        stations = []
        if part_info is None:
            stations.extend(["vgeo1", "hgeo1"])
        else:
            if part_info["rework"]:
                stations.append("rework")
            elif part_info["next_geo"] is not None:
                stations.append(part_info["variant"][0] + "geo" + str(part_info["next_geo"]))
            else:  
                processes = [proc for proc, amount in zip(part_info["proc"], part_info["amount"]) if amount>0]

                for proc in processes:
                    stations.extend(self.part.proc_stat[proc])
        if self.assigned is None or not self.assigned in stations:
            self.assigned = random.choice(stations)
        return self._make_action([self.part.stat_node[self.assigned]], alpyne_obs.caller)


class OnlyStandBehavior(ContextualAgent):
    def __init__(self, hive: HiveContext, n_actions=1) -> None:
        super().__init__(hive)
        self.n_actions = n_actions

    def is_for_learning(self, alpyne_obs: Observation) -> bool:
        super().is_for_learning(alpyne_obs)
        return False

    def get_action(self, alpyne_obs, time):
        return self._make_action(
            [
                0,
            ]
            * self.n_actions,
            alpyne_obs.caller,
        )




class Arc:
    def __init__(self, i, node1, inode1, node2, inode2, maxima) -> None:
        self.i = i
        self.start = inode1
        self.end = inode2
        self.length = np.sqrt(
            np.sum(np.square(np.multiply(np.subtract(node1, node2), maxima)))
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
        while len(H) > 0 and len(H) < 1000:
            H.sort(key=lambda l: l.distance)
            label = H[0]
            if label.arc.start == target_node and label.predecessor != None:
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
        if  len(H) == 0 or len(H) >= 1000:
            print(f"couldn't route from {start_node_coords} to {target_node_coords}")
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
                label.arc.add_block_window(label.start - 2, label.end + 2, who)
                for arc in label.arc.geo_conflicts:
                    arc.add_block_window(label.start - 2, label.end + 2, who)
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
        self.n_stat = len(self.hive.stations)
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
        for i, agv in enumerate(obs["agvs"][np.any(obs["agvs"], axis=1)]):
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
                d["distance_next"] = agv[hwm]
                d["distance_last"] = agv[hwm+1]
                d["invalids"] = agv[hwm+2]
                hwm += 3   
            else:
                d["perc_vb"] = agv[hwm]
                d["perc_hc"] = agv[hwm+1]
                hwm += 2   
            if self.dispatching and not isinstance(self, MultiDispAgent):
                d["part_info"] = self.part_obs_to_dict(agv[hwm:], "matrix" if len(self.hive.nodes)>100 else "minimatrix")
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
        self.actions_len = None

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
        if self.actions_len is None:
            self.actions_len = len(actions)
        if self.dispatching:
            actions = [
                list(self.hive.stations.keys())[action]
                if action is not None and action < len(self.hive.stations)
                else list(self.hive.stations.keys())[0]
                for action in actions
            ] if actions is not None else [list(self.hive.stations.keys())[0]
                for action in range(self.actions_len)]
        else:
            actions = [action if action is not None and  action < 5 else 0 for action in actions] if actions is not None else [0 for _ in range(len(self.action_space.nvec))]
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
        direction_reward = 0,
    ) -> None:
        super().__init__(hive, max_fleetsize, all_ma)
        self.shufflerules = dict()
        self.dispatching = dispatching
        self.start_of_nodes_in_reach = start_of_nodes_in_reach
        self.die_on_target = die_on_target
        self.direction_reward = direction_reward
        if self.direction_reward != 0:
            self.lane_coords_x_1 = [0.23076923076923078,0.893491124260355,0.9526627218934911, 0.4772727272727273]
            self.lane_coords_x_2 = [0.20118343195266272,0.8461538461538461,0.9230769230769231, 0.42045454545454547]
            self.lane_coords_y_1 = [0.5045871559633027, 0.76]
            self.lane_coords_y_2 = [0.5504587155963303, 0.86]

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

        if self.direction_reward != 0:
            if in_obs_caller[2] == in_obs_caller[4]:
                if in_obs_caller[2] in self.lane_coords_x_1 and in_obs_caller[3] < in_obs_caller[5]:
                    reward += self.direction_reward
                if in_obs_caller[2] in self.lane_coords_x_2 and in_obs_caller[3] > in_obs_caller[5]:
                    reward += self.direction_reward
            if in_obs_caller[3] == in_obs_caller[5]:
                if in_obs_caller[3] in self.lane_coords_y_1 and in_obs_caller[2] < in_obs_caller[4]:
                    reward += self.direction_reward
                if in_obs_caller[3] in self.lane_coords_y_2 and in_obs_caller[2] > in_obs_caller[4]:
                    reward += self.direction_reward

        in_obs_agvs = in_obs[other]
        for i in range(len(in_obs_agvs)):
            if in_obs_agvs[i,0] == 0:
                in_obs_agvs[i] = 0
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
            at_target = alpyne_obs.rew >= 0.3
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

class MultiDispAgent(MultiAgent):
    def __init__(self, hive: HiveContext, max_fleetsize, start_of_nodes_in_reach, dispatching=False, all_ma=False, die_on_target=False) -> None:
        super().__init__(hive, max_fleetsize, start_of_nodes_in_reach, dispatching, all_ma, die_on_target)
        self.part = None
        self.env_type = None

    def convert_from_observation(self, alpyne_obs):
        seperate = True
        if self.part is None:
            self.env_type = "matrix" if len(self.hive.stations) > 7 else "minimatrix"
            self.part = MatrixPart() if self.env_type == "matrix" else None
        _obs, _reward, _done, _info =  super().convert_from_observation(alpyne_obs)
        part_obs = _obs["agvs"][:, 18:]

        part_infos = [self.part_obs_to_dict(o, self.env_type) for o in part_obs]
        translated = [self.part.translate(info) if info is not None else None for info in part_infos ]
        new_part_obs = np.zeros(part_obs.shape)
        possible_stations = []
        for i, info in enumerate(translated):
            if info is not None:
                variant = [info["variant"].startswith("vb"), info["variant"].endswith("1")]
                geo_bits = [1 if i+1 == info["next_geo"] else 0 for i in range(self.part.n_geos)]
                proc_values = [v / self.part.max_procedure for v in info["amount"]]
                new_obs = variant + geo_bits + proc_values + [1 if info["rework"] else 0]
                new_part_obs[i, :len(new_obs)] = np.array(new_obs)
                if i == 0:#alpyne_obs.caller - 1000:
                    if info["next_geo"] is not None:
                        possible_stations.append(f'{"v" if info["variant"].startswith("vb") else "h"}geo{info["next_geo"]}')
                    for proc, amount in zip(info["proc"], info["amount"]):
                        if amount > 0:
                            possible_stations.extend(self.part.proc_stat[proc])
                    if info["rework"]:
                        possible_stations.append("rework")
            elif not seperate and i == 0:#alpyne_obs.caller - 1000:
                possible_stations.extend(["vgeo1", "hgeo1"])
        possible_station_nodes = [self.part.stat_node[stat] for stat in possible_stations] + self.part.stat_node["puffer"]
        _obs["agvs"][:, 18:] = new_part_obs
        if not seperate or len(possible_stations)>0:
            _obs["action_mask"] = np.array([1 if stat_node in possible_station_nodes else 0 for stat_node in self.hive.stations.keys()])  
        else:
            _obs["action_mask"] = np.ones((len(self.hive.stations.keys())))
        
        return _obs, _reward, _done, _info

class SingleDispAgent(SingleAgent):
    def __init__(self, hive: HiveContext, max_fleetsize, start_of_nodes_in_reach, dispatching=False) -> None:
        super().__init__(hive, max_fleetsize, start_of_nodes_in_reach, dispatching)
        self.part = None
        self.env_type = None

    def convert_from_observation(self, alpyne_obs):
        if self.part is None:
            self.env_type = "matrix" if len(self.hive.stations) > 7 else "minimatrix"
            self.part = MatrixPart() if self.env_type == "matrix" else None
        _obs, _reward, _done, _info =  super().convert_from_observation(alpyne_obs)
        part_obs = _obs["agvs"][:, 18:]

        part_infos = [self.part_obs_to_dict(o, self.env_type) for o in part_obs]
        translated = [self.part.translate(info) if info is not None else None for info in part_infos ]
        new_part_obs = np.zeros(part_obs.shape)
        possible_stations = {i:[] for i in range(len(translated))}
        for i, info in enumerate(translated):
            if info is not None:
                variant = [info["variant"].startswith("vb"), info["variant"].endswith("1")]
                geo_bits = [1 if i+1 == info["next_geo"] else 0 for i in range(self.part.n_geos)]
                proc_values = [v / self.part.max_procedure for v in info["amount"]]
                new_obs = variant + geo_bits + proc_values + [1 if info["rework"] else 0]
                new_part_obs[i, :len(new_obs)] = np.array(new_obs)
                if i == 0:#alpyne_obs.caller - 1000:
                    if info["next_geo"] is not None:
                        possible_stations[i].append(f'{"v" if info["variant"].startswith("vb") else "h"}geo{info["next_geo"]}')
                    for proc, amount in zip(info["proc"], info["amount"]):
                        if amount > 0:
                            possible_stations[i].extend(self.part.proc_stat[proc])
                    if info["rework"]:
                        possible_stations[i].append("rework")
            else:#alpyne_obs.caller - 1000:
                possible_stations[i].extend(["vgeo1", "hgeo1"])
        possible_station_nodes = {i:[self.part.stat_node[stat] for stat in ps] + self.part.stat_node["puffer"] for i, ps in possible_stations.items()}
        _obs["agvs"][:, 18:] = new_part_obs
        _obs["action_mask"] = np.array([[1 if stat_node in possible_station_nodes[i] else 0 for stat_node in self.hive.stations.keys()] for i in range(len(translated))])
        
        return _obs, _reward, _done, _info


class Dummy(MultiAgent):

    def convert_to_action(self, actions, agent):
        return super().convert_to_action(0, 0)