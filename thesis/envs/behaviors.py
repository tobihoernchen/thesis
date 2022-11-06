from .base_alpyne_zoo import ZooAgentBehavior
from alpyne.data.spaces import Observation, Action
import random
from gym import spaces
import numpy as np


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
            nodes = {i: tuple(n[:2]) for i, n in enumerate(nodecontext)}
            nodes_reversed = {coords: i for i, coords in nodes.items()}
            stations = {i: tuple(n[:2]) for i, n in enumerate(nodecontext) if n[2] == 1}
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
    def __init__(self, n_actions=1) -> None:
        super().__init__()
        self.n_actions = n_actions

    def is_for_learning(self, alpyne_obs: Observation) -> bool:
        super().is_for_learning(alpyne_obs)
        return False

    def get_action(self, alpyne_obs):
        assert super().stations is not None
        actions = random.choices(list(self.stations.keys()), k=self.n_actions)
        return self._make_action(actions, alpyne_obs.caller)


class TrainingBehavior(ContextualAgent):
    observation_space = None
    action_space = None

    def __init__(self, max_fleetsize) -> None:
        super().__init__()
        self.max_fleetsize = max_fleetsize

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
        TrainingBehavior.action_space = spaces.MultiDiscrete(
            [
                self.action_max,
            ]
            * self.max_fleetsize
        )
        TrainingBehavior.observation_space = spaces.Dict(
            {
                "agvs": spaces.Box(0, 1, (self.max_fleetsize, self.obs_shape_len)),
                "stat": spaces.Box(0, 1, (self.n_stat, self.obs_shape_len)),
                "action_mask": spaces.Box(0, 1, (self.max_fleetsize, self.action_max)),
            }
        )

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
            d["last_node"] = agv[1:3]
            d["next_node"] = agv[3:5]
            hwm = 5
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
        super().__init__(max_fleetsize)
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

        obs_action_mask = np.ones((self.max_fleetsize, self.action_max))
        nodes_in_reach = obs_agvs[
            :, self.start_of_nodes_in_reach : self.start_of_nodes_in_reach + 8
        ]
        nodes_in_reach = (nodes_in_reach.reshape(self.max_fleetsize, 4, 2) != 0).any(
            axis=2
        ) * 1
        obs_action_mask[:, 1:] = nodes_in_reach

        obs = {
            "agvs": obs_agvs,
            "stat": obs_stat,
            "action_mask": obs_action_mask,
        }
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
        self, max_fleetsize, start_of_nodes_in_reach, dispatching=False
    ) -> None:
        super().__init__(max_fleetsize)
        self.shufflerules = dict()
        self.dispatching = dispatching
        self.start_of_nodes_in_reach = start_of_nodes_in_reach

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

        obs_action_mask = np.ones((self.max_fleetsize, self.action_max))
        nodes_in_reach = obs_agvs[
            0, self.start_of_nodes_in_reach : self.start_of_nodes_in_reach + 8
        ]
        nodes_in_reach = (nodes_in_reach.reshape(4, 2) != 0).any(axis=1) * 1
        obs_action_mask[0, 1:] = nodes_in_reach

        obs = {
            "agvs": obs_agvs,
            "stat": obs_stat,
            "action_mask": obs_action_mask,
        }
        return obs, reward, False, {"in_system": in_obs_caller[0] == 1}

    def convert_to_action(self, actions, agent):
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


# def get_action_target(self, observation, agent):
#         if isinstance(observation, dict):
#             observation = observation["agvs"]
#         possibles = [
#             observation[x] != 0 and observation[y] != 0
#             for x, y in zip(range(8, 16, 2), range(9, 16, 2))
#         ]
#         next = observation[4:6]
#         target = observation[6:8]
#         action = manual_routing(next, target, possibles)
#         return action, None

#     stations = dict(
#         geo1=(0.32954545454545453, 0.428),
#         geo2=(0.32954545454545453, 0.808),
#         hsn=(0.5727272727272728, 0.62),
#         wps=(0.7909090909090909, 0.62),
#         rework=(0.9727272727272728, 0.62),
#         park=(0.9727272727272728, 1.0),
#     )

#     def get_action_build(self, obs, agent):
#         obs = obs["agvs"]
#         possibles = [
#             obs[x] != 0 and obs[y] != 0
#             for x, y in zip(range(8, 16, 2), range(9, 16, 2))
#         ]
#         var1 = obs[16] == 1
#         var2 = obs[17] == 1
#         geo01 = obs[18] == 1
#         geo12 = obs[19] == 1
#         geo2d = obs[20] == 1
#         d = obs[21] == 1
#         if var1:
#             hwm = 24
#         else:
#             hwm = 26
#         respotsDone = [obs[i] == 1 for i in range(22, hwm, 2)]
#         respotsNio = [obs[i] == 1 for i in range(23, hwm, 2)]
#         if any(respotsNio):
#             self.targets[agent] = "rework"
#         elif not any([geo01, geo12, geo2d, d]):
#             self.targets[agent] = "geo1"
#         elif not any(respotsDone):
#             self.targets[agent] = "wps"
#         elif not all(respotsDone):
#             self.targets[agent] = "hsn"
#         else:
#             self.targets[agent] = "geo2"
#         action = manual_routing(
#             obs[4:6], list(self.stations[self.targets[agent]]), possibles
#         )
#         if action > 0:
#             assert possibles[action - 1]
#         return (
#             action,
#             f"POSSIBLE: {possibles} \t GOAL: {self.targets[agent]}-{list(self.stations[self.targets[agent]])} \t PART:{obs[16:26]}",
#         )

#     def check_possible(self, others_nexts, others_lasts, possibles, my_next):
#         returns = [
#             0,
#         ]
#         for i, possible in enumerate(possibles):
#             is_possible = not possible == (0.0, 0.0)
#             for others_next, others_last in zip(others_nexts, others_lasts):
#                 if possible == others_last and my_next == others_next:
#                     possible = False
#             if is_possible:
#                 returns.append(i + 1)
#         return returns

#     def get_action_rand(self, obs, agent):
#         obs = obs["agvs"]
#         others_lasts = [(obs[i], obs[i + 1]) for i in range(26, len(obs), 24)]
#         others_nexts = [(obs[i], obs[i + 1]) for i in range(28, len(obs), 24)]
#         possibles = [(obs[i], obs[i + 1]) for i in range(6, 14, 2)]
#         possibles = self.check_possible(
#             others_nexts, others_lasts, possibles, (obs[4], obs[5])
#         )
#         if len(possibles) > 1:
#             return random.choice(possibles[1:]), str(possibles) + str(obs[26:30])
#         else:
#             return 0

#     station_nodes = dict(
#         geo1=1,
#         geo2=0,
#         hsn=2,
#         wps=3,
#         rework=4,
#         park=5,
#     )

#     def get_action_disp(self, observation, agent):
#         agent_to_dispatch = agent.split("_")[0]
#         target = self.targets[agent_to_dispatch]
#         if target is None:
#             return random.randint(0, 5), None
#         else:
#             return self.station_nodes[target], target
