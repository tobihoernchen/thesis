import time
import json
import numpy as np
import time
import random


def save_hparams(
    fleetsize=None,
    max_fleetsize=None,
    env_args={},
    algo_args={},
    fe_args={},
    net_arch=[],
    dir="Default",
    run_name="",
    **kwargs,
):
    hparams = dict(
        fleetsize=fleetsize,
        max_fleetsize=max_fleetsize,
        env_args=env_args,
        algo_args=algo_args,
        fe_args=fe_args,
        net_arch=net_arch,
    )
    models_dir = f"../../models/{dir}"  # ../../
    run_name = (
        run_name + f"{fleetsize}-{max_fleetsize}-{time.strftime('%d_%m-%H_%M_%S')}"
    )
    with open(f"{models_dir}/{run_name}.json", "w") as outfile:
        json.dump(hparams, outfile, indent=3)
        return models_dir, run_name


def manual_routing(next, target, possibles=None):
    if possibles is None:
        possibles = [
            True,
        ] * 4

    if np.sqrt(np.square(target[0] - next[0]) + np.square(target[1] - next[1])) < 0.001:
        return 0

    expected = [
        (next[0] - 0.1, next[1]),
        (next[0], next[1] - 0.1),
        (next[0] + 0.1, next[1]),
        (next[0], next[1] + 0.1),
    ]

    distances = [
        np.sqrt(np.square(target[0] - x) + np.square(target[1] - y))
        for x, y in expected
    ]
    possible_distances = [
        distance if possibles[action] else 1e8
        for action, distance in enumerate(distances)
    ]
    return np.argmin(possible_distances) + 1
    # action = None
    # if dx > dy:
    #     if next[0] > target[0]:
    #         action = 1
    #     if next[0] < target[0]:
    #         action = 3
    # if dx < dy:
    #     if next[1] > target[1]:
    #         action = 2
    #     if next[1] < target[1]:
    #         action = 4
    # if action is not None and possibles[action - 1]:
    #     return action
    # elif action is not None:
    #     return random.choice([i + 1 for i, b in enumerate(possibles) if b])
    # else:
    #     return 0


class RewardCheck:
    """WIP"""

    def __init__(self, env, obs2action=None) -> None:
        self.env = env
        self.hist = dict(rewards = [])
        if "agent_selection" in env.__dict__.keys():
            self.zoo = True
            self.gym = False
        else:
            self.zoo = False
            self.gym = True
        if obs2action is not None and not isinstance(obs2action, str):
            self.obs2action = obs2action
        elif obs2action == "target":
            self.obs2action = (
                lambda obs, agent: self.get_action_target(obs, agent)
                if not agent.endswith("Dispatching")
                else self.get_action_disp(obs, agent)
            )
        elif obs2action == "build":
            self.obs2action = (
                lambda obs, agent: self.get_action_build(obs, agent)
                if not agent.endswith("Dispatching")
                else self.get_action_disp(obs, agent)
            )
        else:
            self.obs2action = lambda obs, agent: (
                self.get_action_rand(obs, agent)
                if not agent.endswith("Dispatching")
                else self.get_action_disp(obs, agent)
            )

    def run(
        self,
        n_episodes=None,
        agents=None,
        reward_above=-100,
        reward_below=100,
        seed=None,
    ):
        self.times = []
        counter = 0
        self.reward_above = reward_above
        self.reward_below = reward_below
        self.check_for = (
            agents
            if isinstance(agents, list)
            else [
                agents,
            ]
        )
        while counter < n_episodes or n_episodes is None:
            self.env.reset(seed=seed)
            self.actions = {agent: 0 for agent in self.env.possible_agents}
            self.extras = {agent: None for agent in self.env.possible_agents}
            self.targets = {agent: None for agent in self.env.possible_agents}
            self.total_rew = 0
            if self.gym:
                self.state = [0] * 20, 0, 0, 0
            if self.zoo:
                self.state = {
                    agent: [[0] * 20, 0, 0, 0] for agent in self.env.possible_agents
                }
            self.steps = 0
            self.extra = ""
            if self.zoo:
                for agent in self.env.agent_iter():
                    last = self.env.last()
                    self.hist["rewards"].append(last[1])
                    self.state[agent] = self.check(
                        self.state[agent],
                        self.actions[agent],
                        agent,
                        last,
                        self.extras[agent],
                    )
                    action, extra = self.obs2action(last[0], agent)
                    self.actions[agent] = action
                    self.extras[agent] = extra
                    before = time.time()
                    self.env.step(action)
                    self.times.append(time.time() - before)

                    self.total_rew += self.state[agent][1]
                    self.steps += 1
            if self.gym:
                done = False
                obs = reward = done = info = None
                agent = "default"
                while not done:
                    self.action, self.extra = self.obs2action(
                        obs, reward, done, info, agent
                    )
                    obs, reward, done, info = self.check(
                        self.state,
                        self.action,
                        agent,
                        self.env.step(self.action),
                        self.extra,
                    )
                    self.steps += 1
            counter += 1
            print("+" * 20)
            print(f"Total Reward: {self.total_rew}")
            print(
                f"Mean Step Time: {np.mean(self.times)}s ; Max: {max(self.times)}s ; Min: {min(self.times)}s"
            )

    def check(self, old_state, action, agent, new_state, extra=None):

        obs_o, reward_o, done_o, info_o = old_state
        obs_n, reward_n, done_n, info_n = new_state
        if isinstance(obs_o, dict):
            obs_o = obs_o["agvs"]
        obs_n = obs_n["agvs"]

        if self.check_for[0] is None or agent in self.check_for:
            if reward_n > self.reward_above or reward_n < self.reward_below:
                print(f"{self.steps}---{agent}" + "-" * 20)
                print(f"OLD: {obs_o[:2]} \t done:{done_o}")
                print(f"OLD: curr: {obs_o[2:4]}\t next:{obs_o[4:6]}\t tar:{obs_o[6:8]}")
                print(f"ACTION: {action}")
                print(f"NEW: {obs_n[:2]} \t done:{done_n}")
                print(f"NEW: curr: {obs_n[2:4]}\t next:{obs_n[4:6]}\t tar:{obs_n[6:8]}")
                print(f"REWARD: {reward_n}")
                if extra is not None:
                    print(extra)

        # if action == 1:
        #     assert obs_n[2] >= obs_n[4], (obs_n[2], obs_n[4])
        # if action == 4:
        #     assert obs_n[5] >= obs_n[3], (obs_n[5], obs_n[3])
        # if action == 3:
        #     assert obs_n[4] >= obs_n[2], (obs_n[4], obs_n[2])
        # if action == 2:
        #     assert obs_n[3] >= obs_n[5], (obs_n[3], obs_n[5])
        if np.all(np.isclose(obs_n[4:6], obs_n[6:8], 0.01)):
            pass
            # assert reward_n > 0.7
        if reward_n > 0.7:
            pass
            # assert np.all(np.isclose(obs_n[5:7], obs_n[7:9], 0.01))
        return obs_n, reward_n, done_n, info_n

    def get_action_target(self, observation, agent):
        if isinstance(observation, dict):
            observation = observation["agvs"]
        possibles = [
            observation[x] != 0 and observation[y] != 0
            for x, y in zip(range(8, 16, 2), range(9, 16, 2))
        ]
        next = observation[4:6]
        target = observation[6:8]
        action = manual_routing(next, target, possibles)
        return action, None

    stations = dict(
        geo1=(0.32954545454545453, 0.428),
        geo2=(0.32954545454545453, 0.808),
        hsn=(0.5727272727272728, 0.62),
        wps=(0.7909090909090909, 0.62),
        rework=(0.9727272727272728, 0.62),
        park=(0.9727272727272728, 1.0),
    )

    def get_action_build(self, obs, agent):
        obs = obs["agvs"]
        possibles = [
            obs[x] != 0 and obs[y] != 0
            for x, y in zip(range(8, 16, 2), range(9, 16, 2))
        ]
        var1 = obs[16] == 1
        var2 = obs[17] == 1
        geo01 = obs[18] == 1
        geo12 = obs[19] == 1
        geo2d = obs[20] == 1
        d = obs[21] == 1
        if var1:
            hwm = 24
        else:
            hwm = 26
        respotsDone = [obs[i] == 1 for i in range(22, hwm, 2)]
        respotsNio = [obs[i] == 1 for i in range(23, hwm, 2)]
        if any(respotsNio):
            self.targets[agent] = "rework"
        elif not any([geo01, geo12, geo2d, d]):
            self.targets[agent] = "geo1"
        elif not any(respotsDone):
            self.targets[agent] = "wps"
        elif not all(respotsDone):
            self.targets[agent] = "hsn"
        else:
            self.targets[agent] = "geo2"
        action = manual_routing(
            obs[4:6], list(self.stations[self.targets[agent]]), possibles
        )
        if action > 0:
            assert possibles[action - 1]
        return (
            action,
            f"POSSIBLE: {possibles} \t GOAL: {self.targets[agent]}-{list(self.stations[self.targets[agent]])} \t PART:{obs[16:26]}",
        )

    def check_possible(self, others_nexts, others_lasts, possibles, my_next):
        returns = [
            0,
        ]
        for i, possible in enumerate(possibles):
            is_possible = not possible == (0.0, 0.0)
            for others_next, others_last in zip(others_nexts, others_lasts):
                if possible == others_last and my_next == others_next:
                    possible = False
            if is_possible:
                returns.append(i + 1)
        return returns

    def get_action_rand(self, obs, agent):
        obs = obs["agvs"]
        others_lasts = [(obs[i], obs[i + 1]) for i in range(26, len(obs), 24)]
        others_nexts = [(obs[i], obs[i + 1]) for i in range(28, len(obs), 24)]
        possibles = [(obs[i], obs[i + 1]) for i in range(6, 14, 2)]
        possibles = self.check_possible(
            others_nexts, others_lasts, possibles, (obs[4], obs[5])
        )
        if len(possibles) > 1:
            return random.choice(possibles[1:]), str(possibles) + str(obs[26:30])
        else:
            return 0

    station_nodes = dict(
        geo1=1,
        geo2=0,
        hsn=2,
        wps=3,
        rework=4,
        park=5,
    )

    def get_action_disp(self, observation, agent):
        agent_to_dispatch = agent.split("_")[0]
        target = self.targets[agent_to_dispatch]
        if target is None:
            return random.randint(0, 5), None
        else:
            return self.station_nodes[target], target
